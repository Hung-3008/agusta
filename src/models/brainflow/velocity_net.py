"""VelocityNet — Standard DiT backbone for 1D flow matching.

Architecture follows facebookresearch/DiT with these BrainFlow adaptations:
- Input: 1D fMRI voxels (no patch embedding) → Linear + GELU projection
- Context: multimodal temporal tokens via cross-attention at every DiT block
- Output: per-subject network heads (Schaefer 7-network) or single linear
- Positional encoding: RoPE on context encoder, learned pos_embed on target tokens
"""

import logging
import torch
import torch.nn as nn

from .components import TimestepEmbedder, RotaryEmbedding, RoPETransformerEncoderLayer
from .subject_layers import SubjectLayers, NetworkSubjectLayers
from .fusion import MultiTokenFusion
from .backbones import DiTBackbone, FinalLayer

logger = logging.getLogger(__name__)


class VelocityNet(nn.Module):
    """Velocity network: multitoken context encoder + standard DiT decoder.

    Weight initialization follows the reference DiT exactly:
    - All nn.Linear: xavier_uniform_ with zero bias
    - TimestepEmbedder MLP: normal_(std=0.02)
    - adaLN modulation layers: zero-init (weight and bias)
    - FinalLayer: zero-init (adaLN modulation, linear weight, linear bias)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 768,
        modality_dims: list[int] = None,
        proj_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 12,
        dropout: float = 0.0,
        modality_dropout: float = 0.3,
        max_seq_len: int = 31,
        context_trs: int | None = None,
        n_subjects: int = 4,
        temporal_attn_layers: int = 2,
        fusion_mode: str = "concat",
        fusion_proj_dim: int = 384,
        use_subject_head: bool = True,
        latent_dim: int | None = None,
        gradient_checkpointing: bool = False,
        network_head: bool = False,
        use_rope: bool = False,
        n_target_trs: int = 1,
        context_encoder: str = "multitoken",
        use_dit_decoder: bool = True,
        dit_num_blocks: int | None = None,
        decoder_type: str = "ditx",
        zero_init_network_heads: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [1408]
        self.use_subject_head = use_subject_head
        self.latent_dim = latent_dim if latent_dim is not None else hidden_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.use_rope = use_rope
        self.network_head = network_head and use_subject_head
        self.n_target_trs = n_target_trs
        self.context_encoder = context_encoder
        self.context_trs = int(context_trs) if context_trs is not None else int(max_seq_len)

        if context_encoder != "multitoken":
            raise ValueError(
                "context_encoder must be 'multitoken'. "
                f"Got {context_encoder!r}. Flat encoder was removed in this version."
            )

        dit_depth = dit_num_blocks if dit_num_blocks is not None else n_blocks

        # =====================================================================
        # Context Encoder (unchanged from previous version)
        # =====================================================================

        # Context Fusion
        self.fusion_block = MultiTokenFusion(
            modality_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            modality_dropout=modality_dropout,
            fusion_mode=fusion_mode,
            fusion_proj_dim=fusion_proj_dim,
        )

        # Temporal Encoder
        enc_max_len = max(self.context_trs, max_seq_len)
        if use_rope:
            head_dim = hidden_dim // n_heads
            self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=enc_max_len)
            self.context_pos_emb = None
            self.temporal_attn = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    rotary_emb=self.rotary_emb,
                )
                for _ in range(temporal_attn_layers)
            ])
        else:
            self.context_pos_emb = nn.Parameter(torch.randn(1, enc_max_len, hidden_dim) * 0.02)
            self.rotary_emb = None
            self.temporal_attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=temporal_attn_layers,
            )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # =====================================================================
        # DiT Decoder (standard architecture)
        # =====================================================================

        # Input projection: fMRI voxels → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # Learned positional embeddings on target tokens
        self.target_pos_emb = nn.Parameter(torch.randn(1, n_target_trs, hidden_dim) * 0.02)

        # Timestep embedding (standard DiT: sinusoidal(256) → SiLU MLP → hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # Subject embedding (summed with timestep, like class label in DiT)
        if not use_subject_head:
            self.subject_emb = nn.Embedding(n_subjects, hidden_dim)
        else:
            self.subject_emb = None

        # DiT backbone (stack of DiTCrossBlock)
        dec_max = max(n_target_trs, 64)
        head_dim_d = hidden_dim // n_heads
        self.rotary_emb_decoder = RotaryEmbedding(head_dim_d, max_seq_len=dec_max)
        self.backbone = DiTBackbone(
            hidden_size=hidden_dim,
            num_heads=n_heads,
            depth=dit_depth,
            mlp_ratio=4.0,
            rotary_emb=self.rotary_emb_decoder,
        )
        self.backbone.gradient_checkpointing = self.gradient_checkpointing
        logger.info("Backbone: DiTBackbone (%d blocks, hidden=%d, heads=%d)", dit_depth, hidden_dim, n_heads)

        # FinalLayer (standard DiT: adaLN + zero-init linear)
        if use_subject_head:
            self.final_layer = FinalLayer(hidden_dim, self.latent_dim)
        else:
            self.final_layer = FinalLayer(hidden_dim, output_dim)

        # Subject Heads (per-subject output projection)
        if use_subject_head:
            if self.network_head:
                self.subject_layers = NetworkSubjectLayers(
                    self.latent_dim,
                    n_subjects,
                    zero_init=zero_init_network_heads,
                )
            else:
                self.subject_layers = SubjectLayers(self.latent_dim, output_dim, n_subjects)
        else:
            self.subject_layers = None

        # Initialize weights following DiT reference
        self.initialize_weights()

    def initialize_weights(self):
        """Weight initialization following facebookresearch/DiT exactly."""

        # 1. Global: xavier_uniform_ on all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. TimestepEmbedder MLP: normal_(std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 3. Subject embedding (like label embedding in DiT): normal_(std=0.02)
        if self.subject_emb is not None:
            nn.init.normal_(self.subject_emb.weight, std=0.02)

        # 4. Zero-init adaLN modulation layers in all DiT blocks
        for block in self.backbone.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 5. Zero-init FinalLayer (adaLN modulation + output linear)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def encode_context_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode context: multitoken fusion → temporal encoder → optional slice to ``n_target_trs``."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not getattr(self, "freeze_context", False)):
            splits = []
            offset = 0
            for dim in self.modality_dims:
                splits.append(cond[:, :, offset:offset + dim])
                offset += dim
            context = self.fusion_block(splits)
    
            if self.context_pos_emb is not None:
                Tc = context.shape[1]
                context = context + self.context_pos_emb[:, :Tc, :]
    
            if self.use_rope:
                for layer in self.temporal_attn:
                    if self.gradient_checkpointing and self.training:
                        from torch.utils.checkpoint import checkpoint
                        context = checkpoint(layer, context, use_reentrant=False)
                    else:
                        context = layer(context)
                context = self.temporal_norm(context)
            else:
    
                def _temporal_fwd(x):
                    return self.temporal_norm(self.temporal_attn(x))
    
                if self.gradient_checkpointing and self.training:
                    from torch.utils.checkpoint import checkpoint
                    context = checkpoint(_temporal_fwd, context, use_reentrant=False)
                else:
                    context = _temporal_fwd(context)
    
            slice_start = (self.context_trs - self.n_target_trs) // 2
            context = context[:, slice_start : slice_start + self.n_target_trs, :]
            return context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        pre_encoded_context: torch.Tensor = None,
        subject_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # --- Context ---
        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        elif cond is not None:
            context_encoded = self.encode_context_from_cond(cond)
        else:
            # Unconditional fallback (e.g. CFG with zeroed context)
            tlen = self.n_target_trs
            context_encoded = torch.zeros(
                x.shape[0], tlen, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        # --- Conditioning vector c = t_emb + subject_emb (like DiT: c = t + y) ---
        c = self.t_embedder(t)  # (B, D)
        if self.subject_emb is not None and subject_ids is not None:
            c = c + self.subject_emb(subject_ids)

        # --- Input tokens ---
        h = self.input_proj(x)  # (B, T, D)

        # Add context (additive, like positional encoding) + positional embeddings
        h = h + context_encoded
        T_h = h.shape[1]
        h = h + self.target_pos_emb[:, :T_h, :]

        # --- DiT backbone ---
        h = self.backbone(h, c, context_encoded)  # (B, T, D)

        # --- FinalLayer (adaLN + zero-init linear) ---
        h = self.final_layer(h, c)  # (B, T, latent_dim or output_dim)

        # --- Subject heads ---
        if self.use_subject_head:
            if subject_ids is None:
                subject_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.subject_layers(h, subject_ids)
        return h
