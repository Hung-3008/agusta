"""MoEVAE Temporal Encoder — Hybrid pretrained MoEVAE + Temporal Transformer.

Combines:
  1. Frozen pretrained MultimodalMoEVAE encoder (per-timestep multimodal fusion)
  2. Trainable Temporal Transformer (cross-TR context modeling)

Per TR:  multimodal features → [Frozen MoEVAE] → μ (B, Z_vae)
Stack TRs: (B, T, Z_vae) → [Trainable Temporal Transformer] → (B, T, H)

The output context sequence (B, T, H) feeds into LatentVelocityNet
via cross-attention for conditioning.
"""

import torch
import torch.nn as nn


class MoEVAETemporalEncoder(nn.Module):
    """Hybrid encoder: Frozen MoEVAE (per-TR fusion) + Trainable Temporal Transformer.

    Parameters
    ----------
    moevae : MultimodalMoEVAE
        Pretrained MoEVAE model. Its encoder components (layer_pools, encoders,
        moe_fusion) will be frozen.
    hidden_dim : int
        Hidden dimension of the temporal transformer.
    n_layers : int
        Number of temporal transformer layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for temporal transformer.
    max_seq_len : int
        Maximum sequence length (number of TRs).
    """

    def __init__(
        self,
        moevae: nn.Module,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vae_latent_dim = moevae.latent_dim

        # ---- Frozen MoEVAE encoder components ----
        self.layer_pools = moevae.layer_pools
        self.modality_encoders = moevae.encoders
        self.moe_fusion = moevae.moe_fusion
        self.modality_names = moevae.modality_names

        # Freeze all MoEVAE encoder parts
        for p in self.layer_pools.parameters():
            p.requires_grad = False
        for p in self.modality_encoders.parameters():
            p.requires_grad = False
        for p in self.moe_fusion.parameters():
            p.requires_grad = False

        # ---- Trainable Temporal Transformer ----
        self.input_proj = nn.Sequential(
            nn.Linear(self.vae_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MoEVAETemporalEncoder] frozen={n_frozen:,}, trainable={n_trainable:,}")

    @torch.no_grad()
    def _encode_single_tr(
        self,
        per_tr_features: dict[str, torch.Tensor],  # {mod: (B, D) or (B, L, D)}
    ) -> torch.Tensor:
        """Run frozen MoEVAE encoder on a single TR's features → μ (B, Z_vae)."""
        modality_mus = []

        for mod_name in self.modality_names:
            if mod_name in per_tr_features:
                x = per_tr_features[mod_name]
                # Auto-detect: (B, D) = pre-pooled, (B, L, D) = multi-layer
                if x.dim() == 2:
                    x_pooled = x
                elif x.dim() == 3:
                    x_pooled = self.layer_pools[mod_name](x)
                else:
                    raise ValueError(f"Unexpected dim {x.dim()} for {mod_name}")
                mu_mod, _ = self.modality_encoders[mod_name](x_pooled)
                modality_mus.append(mu_mod)
            else:
                # Missing modality → zero mean
                B = next(iter(per_tr_features.values())).shape[0]
                device = next(iter(per_tr_features.values())).device
                modality_mus.append(
                    torch.zeros(B, self.vae_latent_dim, device=device)
                )

        mu_fused, _ = self.moe_fusion(modality_mus)
        return mu_fused  # (B, Z_vae)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T, D) or (B, T, L, D)}
    ) -> torch.Tensor:
        """Encode multimodal features → context sequence (B, T, H).

        Args:
            modality_features: dict of modality features.
                Each value has shape (B, T, D) for pre-pooled
                or (B, T, n_layers, D) for multilayer.

        Returns:
            context: (B, T, H) context sequence for cross-attention.
        """
        # Determine B, T from first modality
        first_feat = next(iter(modality_features.values()))
        B = first_feat.shape[0]
        T = first_feat.shape[1]

        # ---- Frozen: Encode each TR through MoEVAE ----
        tr_embeddings = []
        for t in range(T):
            # Extract per-TR features
            per_tr = {}
            for mod_name, feat in modality_features.items():
                if feat.dim() == 3:
                    per_tr[mod_name] = feat[:, t, :]       # (B, D)
                elif feat.dim() == 4:
                    per_tr[mod_name] = feat[:, t, :, :]    # (B, L, D)
                else:
                    raise ValueError(f"Unexpected shape {feat.shape} for {mod_name}")

            mu_t = self._encode_single_tr(per_tr)  # (B, Z_vae)
            tr_embeddings.append(mu_t)

        z_seq = torch.stack(tr_embeddings, dim=1)  # (B, T, Z_vae)

        # ---- Trainable: Temporal Transformer ----
        h = self.input_proj(z_seq)                      # (B, T, H)
        h = h + self.pos_embed[:, :T, :]                # + positional embedding
        h = self.temporal_transformer(h)                 # (B, T, H)
        h = self.final_norm(h)                           # (B, T, H)

        return h


class MoEVAEDirectEncoder(nn.Module):
    """Direct MoEVAE encoder: Frozen MoEVAE per-TR, no temporal transformer.

    Variant A: minimal trainable params. Relies on VelocityNet's cross-attention
    to implicitly learn temporal patterns from the sequence of MoEVAE embeddings.

    Per TR:  multimodal features → [Frozen MoEVAE] → μ (B, Z_vae=512)
    Stack TRs → (B, T, Z_vae) — fed directly to VelocityNet cross-attention.

    Parameters
    ----------
    moevae : MultimodalMoEVAE
        Pretrained MoEVAE model. Encoder parts will be frozen.
    """

    def __init__(self, moevae: nn.Module):
        super().__init__()
        self.hidden_dim = moevae.latent_dim  # e.g., 512
        self.vae_latent_dim = moevae.latent_dim

        # ---- Frozen MoEVAE encoder components ----
        self.layer_pools = moevae.layer_pools
        self.modality_encoders = moevae.encoders
        self.moe_fusion = moevae.moe_fusion
        self.modality_names = moevae.modality_names

        # Freeze all MoEVAE encoder parts
        for p in self.layer_pools.parameters():
            p.requires_grad = False
        for p in self.modality_encoders.parameters():
            p.requires_grad = False
        for p in self.moe_fusion.parameters():
            p.requires_grad = False

        n_frozen = sum(p.numel() for p in self.parameters())
        print(f"[MoEVAEDirectEncoder] frozen={n_frozen:,}, trainable=0, "
              f"output_dim={self.hidden_dim}")

    @torch.no_grad()
    def _encode_single_tr(
        self,
        per_tr_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run frozen MoEVAE encoder on a single TR's features → μ (B, Z_vae)."""
        modality_mus = []

        for mod_name in self.modality_names:
            if mod_name in per_tr_features:
                x = per_tr_features[mod_name]
                if x.dim() == 2:
                    x_pooled = x
                elif x.dim() == 3:
                    x_pooled = self.layer_pools[mod_name](x)
                else:
                    raise ValueError(f"Unexpected dim {x.dim()} for {mod_name}")
                mu_mod, _ = self.modality_encoders[mod_name](x_pooled)
                modality_mus.append(mu_mod)
            else:
                B = next(iter(per_tr_features.values())).shape[0]
                device = next(iter(per_tr_features.values())).device
                modality_mus.append(
                    torch.zeros(B, self.vae_latent_dim, device=device)
                )

        mu_fused, _ = self.moe_fusion(modality_mus)
        return mu_fused

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode multimodal features → context sequence (B, T, Z_vae).

        No temporal transformer — raw MoEVAE embeddings per TR.
        VelocityNet cross-attention handles temporal modeling.
        """
        first_feat = next(iter(modality_features.values()))
        B = first_feat.shape[0]
        T = first_feat.shape[1]

        tr_embeddings = []
        for t in range(T):
            per_tr = {}
            for mod_name, feat in modality_features.items():
                if feat.dim() == 3:
                    per_tr[mod_name] = feat[:, t, :]
                elif feat.dim() == 4:
                    per_tr[mod_name] = feat[:, t, :, :]
                else:
                    raise ValueError(f"Unexpected shape {feat.shape} for {mod_name}")

            mu_t = self._encode_single_tr(per_tr)
            tr_embeddings.append(mu_t)

        z_seq = torch.stack(tr_embeddings, dim=1)  # (B, T, Z_vae)
        return z_seq


def _load_moevae(moevae_checkpoint, modality_configs, moevae_params, device):
    """Shared helper: build and load MoEVAE from checkpoint."""
    from src.models.brainflow.multimodal_vae import MultimodalMoEVAE

    moevae_p = moevae_params or {}

    moevae = MultimodalMoEVAE(
        modality_configs=modality_configs,
        latent_dim=moevae_p.get("latent_dim", 512),
        encoder_hidden=moevae_p.get("encoder_hidden", 1024),
        n_experts=moevae_p.get("n_experts", 4),
        expert_hidden=moevae_p.get("expert_hidden", 1024),
        decoder_hidden=moevae_p.get("decoder_hidden", 1024),
        dropout=moevae_p.get("dropout", 0.1),
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(moevae_checkpoint, map_location=device, weights_only=False)
    if "ema_model" in ckpt:
        moevae.load_state_dict(ckpt["ema_model"])
        print(f"[load_moevae] Loaded from EMA weights")
    elif "model" in ckpt:
        moevae.load_state_dict(ckpt["model"])
        print(f"[load_moevae] Loaded from model weights")
    else:
        moevae.load_state_dict(ckpt)
        print(f"[load_moevae] Loaded state dict directly")

    return moevae


def build_moevae_direct_encoder(
    moevae_checkpoint: str,
    modality_configs: dict,
    moevae_params: dict = None,
    device: torch.device = None,
) -> MoEVAEDirectEncoder:
    """Build MoEVAEDirectEncoder (Variant A: no temporal transformer).

    Parameters
    ----------
    moevae_checkpoint : str
        Path to pretrained MoEVAE checkpoint.
    modality_configs : dict
        {mod_name: {'dim': int, 'n_layers': int}}.
    moevae_params : dict
        MoEVAE model params.
    device : torch.device
    """
    moevae = _load_moevae(moevae_checkpoint, modality_configs, moevae_params, device)
    return MoEVAEDirectEncoder(moevae)


def build_moevae_temporal_encoder(
    moevae_checkpoint: str,
    modality_configs: dict,
    moevae_params: dict = None,
    temporal_params: dict = None,
    device: torch.device = None,
) -> MoEVAETemporalEncoder:
    """Build MoEVAETemporalEncoder (Variant B: with temporal transformer).

    Parameters
    ----------
    moevae_checkpoint : str
        Path to pretrained MoEVAE checkpoint.
    modality_configs : dict
        {mod_name: {'dim': int, 'n_layers': int}}.
    moevae_params : dict
        MoEVAE model params.
    temporal_params : dict
        Temporal transformer params (hidden_dim, n_layers, n_heads, dropout).
    device : torch.device
    """
    temporal_p = temporal_params or {}
    moevae = _load_moevae(moevae_checkpoint, modality_configs, moevae_params, device)

    return MoEVAETemporalEncoder(
        moevae=moevae,
        hidden_dim=temporal_p.get("hidden_dim", 1024),
        n_layers=temporal_p.get("n_layers", 4),
        n_heads=temporal_p.get("n_heads", 8),
        dropout=temporal_p.get("dropout", 0.1),
        max_seq_len=temporal_p.get("max_seq_len", 64),
    )

