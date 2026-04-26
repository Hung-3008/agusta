import torch
import torch.nn as nn

class MultiTokenFusion(nn.Module):
    """Per-modality projection + embeddings, then mean or concat (TRIBE-style) fusion.

    ``fusion_mode="mean"``: project each modality to ``hidden_dim``, mean pool,
    ``output_proj`` (legacy).

    ``fusion_mode="concat"``: project each to ``fusion_proj_dim``, add modality_emb
    in that space, concatenate on the feature axis → ``(B,T,M*P)``, then
    ``fusion_to_hidden`` → ``(B,T,hidden_dim)``.
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 1024,
        proj_dim: int = 256,
        max_seq_len: int = 11,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        fusion_mode: str = "concat",
        fusion_proj_dim: int = 384,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.max_seq_len = max_seq_len
        self.modality_dropout = modality_dropout
        self.fusion_mode = fusion_mode
        self.fusion_proj_dim = fusion_proj_dim

        if fusion_mode not in ("mean", "concat"):
            raise ValueError(f"fusion_mode must be 'mean' or 'concat', got {fusion_mode!r}")

        if fusion_mode == "mean":
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for dim in modality_dims
            ])
            self.modality_emb = nn.Parameter(
                torch.randn(self.n_modalities, hidden_dim) * 0.02
            )
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fusion_to_hidden = None
            self.concat_dim = None
        else:
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, fusion_proj_dim),
                    nn.LayerNorm(fusion_proj_dim),
                    nn.GELU(),
                )
                for dim in modality_dims
            ])
            self.modality_emb = nn.Parameter(
                torch.randn(self.n_modalities, fusion_proj_dim) * 0.02
            )
            self.concat_dim = self.n_modalities * fusion_proj_dim
            intermediate_dim = 2048
            self.fusion_to_hidden = nn.Sequential(
                nn.Linear(self.concat_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output_proj = None

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: List of M tensors, each (B, T, mod_dim_i).

        Returns:
            context: (B, T, hidden_dim) fused context.
        """
        B, T = modality_features[0].shape[:2]
        T = min(T, self.max_seq_len)

        projected = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.projectors)):
            h = proj(feat[:, :T])
            h = h + self.modality_emb[i]
            projected.append(h)

        if self.training and self.modality_dropout > 0:
            keep_mask = (
                torch.rand(B, 1, self.n_modalities, device=projected[0].device)
                > self.modality_dropout
            )
            all_dropped = (keep_mask.sum(dim=2, keepdim=True) == 0)
            keep_mask[:, :, 0:1] = torch.max(keep_mask[:, :, 0:1], all_dropped)

            for i in range(self.n_modalities):
                projected[i] = projected[i] * keep_mask[:, :, i:i+1]

        if self.fusion_mode == "mean":
            x = torch.stack(projected, dim=0).mean(dim=0)
            return self.output_proj(x)

        x = torch.cat(projected, dim=-1)
        return self.fusion_to_hidden(x)
