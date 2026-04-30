import logging
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .components import SinusoidalPosEmb, RotaryEmbedding, RoPETransformerEncoderLayer
from .subject_layers import build_subject_head
from .fusion import MultiTokenFusion
from .backbones import DiTXBackbone, DiT1DBackbone
from .mamba_backbone import MambaFlowBackbone

logger = logging.getLogger(__name__)


class VelocityNet(nn.Module):
    """Velocity network with multitoken context encoder, optional temporal slice, and plug-and-play backbone."""

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        modality_dims: list[int] = None,
        proj_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
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
        subject_head_type: str = "linear",
        subject_head_hidden_mult: float = 1.0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
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
        self.subject_head_type = subject_head_type
        self.subject_head_hidden_mult = subject_head_hidden_mult
        self.n_target_trs = n_target_trs
        self.context_encoder = context_encoder
        self.context_trs = int(context_trs) if context_trs is not None else int(max_seq_len)

        if context_encoder != "multitoken":
            raise ValueError(
                "context_encoder must be 'multitoken'. "
                f"Got {context_encoder!r}. Flat encoder was removed in this version."
            )

        self._decoder_type = decoder_type
        dit_depth = dit_num_blocks if dit_num_blocks is not None else n_blocks

        # Learned positional embeddings on target tokens
        if decoder_type == "ditx":
            self.target_pos_emb = nn.Parameter(torch.randn(1, n_target_trs, hidden_dim) * 0.02)
        else:
            self.target_pos_emb = None

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

        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # Time Embeddings
        self.time_embed = SinusoidalPosEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Subject Heads
        if use_subject_head:
            self.subject_layers = build_subject_head(
                in_channels=self.latent_dim,
                output_dim=output_dim,
                n_subjects=n_subjects,
                network_head=self.network_head,
                zero_init=zero_init_network_heads,
                head_type=subject_head_type,
                hidden_mult=subject_head_hidden_mult,
            )
        else:
            self.subject_layers = None

        # Always create subject embedding so DiT backbone gets subject
        # conditioning via AdaLN-Zero (shift/scale/gate become subject-aware).
        self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        # Modular Backbone
        dec_max = max(n_target_trs, 64)
        head_dim_d = hidden_dim // n_heads
        self.rotary_emb_decoder = RotaryEmbedding(head_dim_d, max_seq_len=dec_max)
        if decoder_type == "ditx":
            self.backbone = DiTXBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth
            )
            logger.info("Backbone: DiTXBackbone (%d blocks)", dit_depth)
        elif decoder_type == "mamba":
            self.backbone = MambaFlowBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand_factor=mamba_expand,
            )
            logger.info("Backbone: MambaFlowBackbone (%d blocks, d_state=%d, expand=%d)",
                        dit_depth, mamba_d_state, mamba_expand)
        else:
            self.backbone = DiT1DBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth
            )
            logger.info("Backbone: DiT1DBackbone (%d blocks)", dit_depth)
        self.backbone.gradient_checkpointing = self.gradient_checkpointing

        # Output Layer
        self.final_norm = nn.LayerNorm(hidden_dim)
        if use_subject_head:
            self.latent_head = nn.Linear(hidden_dim, self.latent_dim)
            self.output_layer = None
        else:
            self.latent_head = None
            self.output_layer = nn.Linear(hidden_dim, output_dim)
            nn.init.constant_(self.output_layer.weight, 0)
            nn.init.constant_(self.output_layer.bias, 0)

    def encode_context_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode context: multitoken fusion → temporal encoder → optional slice to ``n_target_trs``."""
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
                    context = checkpoint(layer, context, use_reentrant=False)
                else:
                    context = layer(context)
            context = self.temporal_norm(context)
        else:

            def _temporal_fwd(x):
                return self.temporal_norm(self.temporal_attn(x))

            if self.gradient_checkpointing and self.training:
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

        t_emb = self.time_mlp(self.time_embed(t))

        if subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        h = self.input_proj(x)

        # Prepare tokens for backbone
        h = h + context_encoded
        if self.target_pos_emb is not None:
            T_h = h.shape[1]
            h = h + self.target_pos_emb[:, :T_h, :]

        # Delegate to plug-and-play Backbone
        h = self.backbone(h, t_emb, context_encoded)

        h = self.final_norm(h)
        if self.use_subject_head:
            z = self.latent_head(h)
            if subject_ids is None:
                subject_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.subject_layers(z, subject_ids)
        return self.output_layer(h)
