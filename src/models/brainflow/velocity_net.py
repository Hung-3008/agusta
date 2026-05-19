import logging
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .components import SinusoidalPosEmb, RotaryEmbedding, RoPETransformerEncoderLayer
from .subject_layers import SubjectLayers, NetworkSubjectLayers, VoxelPersonalityHead
from .fusion import MultiTokenFusion
from .backbones import DiTXBackbone, DiT1DBackbone, DiTOriginalBackbone, DiTHybridBackbone, DiTJointBackbone, MLPBackbone, UDiT1DBackbone

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
        cross_attn_every_n: int = 4,
        stochastic_depth_rate: float = 0.0,
        mlp_depth: int = 8,
        head_type: str = "auto",
        voxel_personality_dim: int = 128,
        voxel_personality_bias: bool = True,
        subject_adaln: bool = True,
        # U-DiT specific params
        udit_depth: list[int] = None,
        udit_channel_mult: list[float] = None,
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

        # Subject Heads (B1 — head selection)
        # head_type:
        #   "auto"               — legacy: use SubjectLayers / NetworkSubjectLayers (based on network_head)
        #   "flat"               — force SubjectLayers
        #   "network"            — force NetworkSubjectLayers
        #   "voxel_personality"  — VoxelPersonalityHead (bilinear factorization, B1)
        self.head_type = head_type if head_type != "auto" else (
            "network" if self.network_head else "flat"
        )
        if use_subject_head:
            if self.head_type == "voxel_personality":
                self.subject_layers = VoxelPersonalityHead(
                    latent_dim=self.latent_dim,
                    n_voxels=output_dim,
                    n_subjects=n_subjects,
                    voxel_dim=voxel_personality_dim,
                    bias=voxel_personality_bias,
                    zero_init_w=True,
                )
            elif self.head_type == "network":
                self.subject_layers = NetworkSubjectLayers(
                    self.latent_dim,
                    n_subjects,
                    zero_init=zero_init_network_heads,
                )
            else:
                self.subject_layers = SubjectLayers(self.latent_dim, output_dim, n_subjects)
        else:
            self.subject_layers = None

        # B2 — Subject AdaLN: always create subject embedding so DiT trunk
        # receives subject conditioning via t_emb. Zero-init when a subject
        # head is present so warm-starting from old checkpoints is identity-safe.
        if subject_adaln:
            self.subject_emb = nn.Embedding(n_subjects, hidden_dim)
            if use_subject_head:
                nn.init.zeros_(self.subject_emb.weight)
        else:
            self.subject_emb = None

        # Modular Backbone
        dec_max = max(n_target_trs, 64)
        head_dim_d = hidden_dim // n_heads
        self.rotary_emb_decoder = RotaryEmbedding(head_dim_d, max_seq_len=dec_max)
        if decoder_type == "ditx":
            self.backbone = DiTXBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth, stochastic_depth_rate=stochastic_depth_rate,
            )
            logger.info("Backbone: DiTXBackbone (%d blocks, stoch_depth=%.2f)",
                        dit_depth, stochastic_depth_rate)
        elif decoder_type == "dit_original":
            self.backbone = DiTOriginalBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth
            )
            logger.info("Backbone: DiTOriginalBackbone (%d blocks)", dit_depth)
        elif decoder_type == "dit_hybrid":
            self.backbone = DiTHybridBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth, cross_attn_every_n=cross_attn_every_n,
                stochastic_depth_rate=stochastic_depth_rate,
            )
            n_ca = sum(1 for i in range(dit_depth) if (i + 1) % cross_attn_every_n == 0)
            logger.info("Backbone: DiTHybridBackbone (%d blocks, %d cross-attn every %d)",
                        dit_depth, n_ca, cross_attn_every_n)
        elif decoder_type == "dit_joint":
            self.backbone = DiTJointBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                dit_depth=dit_depth
            )
            logger.info("Backbone: DiTJointBackbone (MMDiT, %d blocks, bidirectional context)", dit_depth)
        elif decoder_type == "udit":
            _udit_depth = udit_depth if udit_depth is not None else [2, 4, 6, 4, 2]
            _udit_ch_mult = udit_channel_mult if udit_channel_mult is not None else [1, 2, 4]
            self.backbone = UDiT1DBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                udit_depth=_udit_depth,
                channel_mult=_udit_ch_mult,
                cross_attn_every_n=cross_attn_every_n,
                stochastic_depth_rate=stochastic_depth_rate,
                n_target_trs=n_target_trs,
            )
            _total_blocks = sum(_udit_depth)
            logger.info(
                "Backbone: UDiT1DBackbone (depth=%s, ch_mult=%s, %d total blocks, cross_attn_every=%d)",
                _udit_depth, _udit_ch_mult, _total_blocks, cross_attn_every_n,
            )
        elif decoder_type == "mlp":
            self.backbone = MLPBackbone(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, time_dim=hidden_dim, rotary_emb=self.rotary_emb_decoder,
                mlp_depth=mlp_depth,
            )
            logger.info("Backbone: MLPBackbone (%d layers, no attention)", mlp_depth)
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

        # B2 — inject subject conditioning into AdaLN modulation of every DiT block.
        # Active for both use_subject_head=True (zero-init → warm-start safe) and =False.
        if self.subject_emb is not None and subject_ids is not None:
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
