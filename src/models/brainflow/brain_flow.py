"""
BrainFlow — FlowFM-Style Direct Flow Matching for fMRI Generation.

Architecture:
  Multimodal Features → Transformer Encoder → cls_token (B, D_enc) + patch_tokens
  Timestep t → sinusoidal_embed → t_emb (B, D)
  c = CondMLP(concat[proj(cls_token), t_emb]) → (B, D)

  x_t (B, n_voxels) → patchify → patch_embed → (B, num_patches, D)
  → DiT Blocks × N (self-attn + cross-attn + FFN, adaLN-Zero from c)
  → output_proj → unpatchify → v_pred (B, n_voxels)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Utilities ────────────────────────────────────────────────────────────────


def drop_path(x, drop_prob: float = 0., training: bool = False,
              scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─── Subject-Specific Layers ─────────────────────────────────────────────────


class SubjectLayers(nn.Module):
    """Per-subject linear projection (adapted from TRIDE).

    Maintains separate weight matrices for each subject, enabling
    subject-specific mapping from shared representation to brain space.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output dimension (e.g., patch_size).
    n_subjects : int
        Number of subjects.
    bias : bool
        Whether to include a per-subject bias.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 n_subjects: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(n_subjects, in_channels, out_channels))
        self.bias = (
            nn.Parameter(torch.empty(n_subjects, out_channels))
            if bias else None
        )
        # Init: scaled normal (same as TRIDE)
        self.weights.data.normal_()
        self.weights.data *= 1 / in_channels ** 0.5
        if self.bias is not None:
            self.bias.data.normal_()
            self.bias.data *= 1 / in_channels ** 0.5

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """Apply subject-specific linear projection.

        Args:
            x: (B, N, C_in) — input tokens
            subjects: (B,) — integer subject indices

        Returns:
            (B, N, C_out)
        """
        weights = self.weights[subjects]      # (B, C_in, C_out)
        out = torch.bmm(x, weights)           # (B, N, C_out)
        if self.bias is not None:
            out = out + self.bias[subjects].unsqueeze(1)  # (B, 1, C_out)
        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, n_subjects={S})"


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BrainFlowConfig:
    # fMRI dimensions
    n_voxels: int = 1000
    patch_size: int = 100

    # Transformer (velocity network)
    hidden_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # Multimodal Feature Encoder
    feature_dims: Dict[str, int] = field(default_factory=dict)
    encoder_dim: int = 1024
    encoder_depth: int = 2
    encoder_heads: int = 8
    feature_patches: int = 1   # Split each non-omni feature into P patches/TR

    # Conditioning
    cond_dim: int = 512
    use_cross_attention: bool = True

    # Subject-specific
    n_subjects: int = 4


# ─── Multimodal Feature Encoder ──────────────────────────────────────────────


class MultimodalFeatureEncoder(nn.Module):
    """
    Trainable Multimodal Feature Encoder f_φ.

    Input:  dict of features e.g. {"video": (B, T, D_v), "omni": (B, T*N, D_o)}
    Output: cls_token (B, D_encoder), patch_tokens (B, N_total, D_encoder)

    Feature Patchification:
      For non-omni modalities, each feature vector D is split into P patches
      of size D//P, creating P tokens per TR.
    """

    def __init__(self, config: BrainFlowConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.encoder_dim
        P = config.feature_patches

        # Track how many patches each modality produces per TR
        self.patches_per_modality: Dict[str, int] = {}

        # Projections for each modality to common embed_dim
        self.projections = nn.ModuleDict()
        for modality, dim in config.feature_dims.items():
            if modality == "omni":
                self.projections[modality] = nn.Linear(dim, self.embed_dim)
                self.patches_per_modality[modality] = 1
            else:
                patch_dim = dim // P
                if dim % P != 0:
                    raise ValueError(
                        f"Feature dim {dim} for '{modality}' is not divisible "
                        f"by feature_patches={P}. Choose P that divides all dims."
                    )
                self.projections[modality] = nn.Linear(patch_dim, self.embed_dim)
                self.patches_per_modality[modality] = P

        # Cross-modal Transformer Encoder
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=config.encoder_heads,
                dim_feedforward=self.embed_dim * 4,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=config.encoder_depth)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Modality embeddings
        self.modality_embeds = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
            for modality in config.feature_dims.keys()
        })

        # Final LayerNorm on representation
        self.repr_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, features_dict):
        """
        features_dict: dict of tensors
            Non-omni: (B, T, D) where T = context_trs
            Omni: (B, T*N_tokens, D)
        Returns:
            cls_token: (B, embed_dim)
            patch_tokens: (B, N_total, embed_dim)
        """
        B = next(iter(features_dict.values())).shape[0]

        projected_seqs = []
        for modality, x in features_dict.items():
            if modality not in self.projections:
                continue

            P = self.patches_per_modality[modality]

            if P > 1:
                # Feature Patchification: (B, T, D) → (B, T*P, D//P)
                B_, T, D = x.shape
                x = x.reshape(B_, T * P, D // P)

            x_proj = self.projections[modality](x)
            x_proj = x_proj + self.modality_embeds[modality]
            projected_seqs.append(x_proj)

        if not projected_seqs:
            raise ValueError("No matching modalities found in features_dict")

        # Concatenate along token dimension
        seq = torch.cat(projected_seqs, dim=1)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat((cls_tokens, seq), dim=1)

        # Pass through transformer
        out_seq = self.transformer(seq)

        cls_token = out_seq[:, 0]
        patch_tokens = out_seq[:, 1:]

        cls_token = self.repr_norm(cls_token)
        return cls_token, patch_tokens


# ─── FlowFM-style Conditioning MLP ──────────────────────────────────────────


class ConditioningMLP(nn.Module):
    """
    FlowFM-style conditioning: merges representation r and timestep t
    into a single conditioning vector c for adaLN-Zero.

    c = MLP(concat[proj(r), t_emb])
    """

    def __init__(self, encoder_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()

        self.repr_proj = nn.Linear(encoder_dim, cond_dim)

        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.merge_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim),
        )

    def forward(self, r, t_emb):
        """
        r: (B, encoder_dim) — representation from encoder
        t_emb: (B, hidden_dim) — sinusoidal timestep embedding

        Returns: c (B, hidden_dim) — condition vector for adaLN
        """
        r_proj = self.repr_proj(r)
        t_proj = self.t_mlp(t_emb)
        merged = torch.cat([r_proj, t_proj], dim=-1)
        c = self.merge_mlp(merged)
        return c


# ─── FlowFM-style DiT Block ─────────────────────────────────────────────────


class FlowFMDiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning + optional cross-attention.

    Order: Self-Attn → Cross-Attn (optional) → FFN
    All sublayers use adaLN-Zero modulation from condition vector c.
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0,
                 dropout=0.0, drop_path_rate=0.0,
                 use_cross_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate
        self.use_cross_attention = use_cross_attention

        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-Attention (fMRI tokens attend to encoder patch tokens)
        if use_cross_attention:
            self.cross_norm = nn.LayerNorm(
                hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # adaLN-Zero: 9 params if cross-attn, else 6
        n_mod = 9 if use_cross_attention else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, n_mod * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, context=None):
        """
        x: (B, N, D) — fMRI tokens
        c: (B, D) — condition vector
        context: (B, M, D) — encoder patch tokens (for cross-attn)
        """
        mod = self.adaLN_modulation(c)

        if self.use_cross_attention:
            (shift1, scale1, gate1,
             shift_ca, scale_ca, gate_ca,
             shift2, scale2, gate2) = mod.chunk(9, dim=-1)
        else:
            shift1, scale1, gate1, shift2, scale2, gate2 = \
                mod.chunk(6, dim=-1)

        # 1) Self-Attention with adaLN
        h1 = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + drop_path(
            gate1.unsqueeze(1) * attn_out,
            self.drop_path_rate, self.training)

        # 2) Cross-Attention with adaLN
        if self.use_cross_attention and context is not None:
            h_ca = modulate(self.cross_norm(x), shift_ca, scale_ca)
            ca_out, _ = self.cross_attn(h_ca, context, context)
            x = x + drop_path(
                gate_ca.unsqueeze(1) * ca_out,
                self.drop_path_rate, self.training)

        # 3) FFN with adaLN
        h2 = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h2)
        x = x + drop_path(
            gate2.unsqueeze(1) * mlp_out,
            self.drop_path_rate, self.training)

        return x


# ─── Main Model ──────────────────────────────────────────────────────────────


class BrainFlow(nn.Module):
    """
    BrainFlow — FlowFM-style Direct Flow Matching for fMRI.

    Multimodal features → Transformer encoder → adaLN-Zero conditioning
    → DiT velocity network → predicted velocity field.
    """

    def __init__(self, config: Optional[BrainFlowConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BrainFlowConfig(**kwargs)
        self.config = config
        D = config.hidden_dim

        # ── Representation Encoder f_φ ──
        self.encoder = MultimodalFeatureEncoder(config)

        # ── Conditioning MLP (CLS token + timestep → adaLN) ──
        self.cond_mlp = ConditioningMLP(
            encoder_dim=self.encoder.embed_dim,
            hidden_dim=D,
            cond_dim=config.cond_dim,
        )

        # ── Context projection (patch tokens → cross-attn K/V) ──
        self.use_cross_attention = config.use_cross_attention
        if self.use_cross_attention:
            self.context_proj = nn.Linear(self.encoder.embed_dim, D)
            self.context_norm = nn.LayerNorm(D)

        # ── fMRI Patchification ──
        if config.n_voxels % config.patch_size != 0:
            self.padded_voxels = (
                (config.n_voxels // config.patch_size) + 1
            ) * config.patch_size
        else:
            self.padded_voxels = config.n_voxels
        self.num_patches = self.padded_voxels // config.patch_size
        self.pad_len = self.padded_voxels - config.n_voxels

        # Patch embedding: patch_size → D
        self.patch_embed = nn.Linear(config.patch_size, D)

        # ── Positional Embedding ──
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, D) * 0.02)

        # ── DiT Blocks ──
        dpr = [x.item() for x in torch.linspace(
            0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            FlowFMDiTBlock(
                hidden_dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path_rate=dpr[i],
                use_cross_attention=config.use_cross_attention,
            )
            for i in range(config.depth)
        ])

        # ── Output Head ──
        self.final_layer_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(D, 2 * D))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        # Subject-specific output projection
        self.output_proj = SubjectLayers(
            in_channels=D, out_channels=config.patch_size,
            n_subjects=config.n_subjects, bias=True,
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for velocity network (not encoder, not SubjectLayers)."""
        for name, m in self.named_modules():
            if name.startswith('encoder'):
                continue
            if isinstance(m, SubjectLayers):
                continue  # SubjectLayers has its own init
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Re-zero adaLN outputs for zero-init residual
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

    def _patchify(self, x):
        """(B, n_voxels) → (B, num_patches, patch_size)"""
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        return x.view(x.shape[0], self.num_patches, self.config.patch_size)

    def _unpatchify(self, x):
        """(B, num_patches, patch_size) → (B, n_voxels)"""
        x = x.reshape(x.shape[0], -1)
        if self.pad_len > 0:
            x = x[:, :self.config.n_voxels]
        return x

    def _encode_and_condition(self, t, features_dict, drop_repr=False,
                               drop_mask=None):
        """Shared encoder + conditioning logic.

        Args:
            t: (B,) timesteps
            features_dict: multimodal features
            drop_repr: if True, zero ALL representations (for CFG uncond)
            drop_mask: (B,) bool tensor, zero representations per-sample (DGS)

        Returns:
            c: (B, D) conditioning vector
            context: (B, M, D) or None, cross-attention context
        """
        cls_token, patch_tokens = self.encoder(features_dict)

        # DGS: per-sample or full dropout of representation
        if drop_repr:
            cls_token = torch.zeros_like(cls_token)
            patch_tokens = torch.zeros_like(patch_tokens)
        elif drop_mask is not None and drop_mask.any():
            cls_token = cls_token.clone()
            patch_tokens = patch_tokens.clone()
            cls_token[drop_mask] = 0.0
            patch_tokens[drop_mask] = 0.0

        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        c = self.cond_mlp(cls_token, t_emb)

        context = None
        if self.use_cross_attention:
            context = self.context_norm(self.context_proj(patch_tokens))

        return c, context

    def _velocity_head(self, x_t, c, context, subject_id):
        """Shared velocity network: patchify → DiT blocks → unpatchify.

        Args:
            x_t: (B, n_voxels)
            c: (B, D) conditioning
            context: (B, M, D) or None
            subject_id: (B,) integer subject indices

        Returns:
            v_pred: (B, n_voxels)
        """
        x_patches = self._patchify(x_t)
        x_tokens = self.patch_embed(x_patches) + self.patch_pos_embed

        for block in self.blocks:
            x_tokens = block(x_tokens, c, context)

        mod_params = self.final_adaLN(c)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_tokens), shift, scale)
        x_out = self.output_proj(x_out, subject_id)

        return self._unpatchify(x_out)

    def forward(self, t, x_t, features_dict, subject_id, drop_repr=False):
        """
        Predict velocity v(x_t, t | features_dict).

        Args:
            t: (B,) timestep values in [0, 1]
            x_t: (B, n_voxels) point on the probability path
            features_dict: dict of input features
            subject_id: (B,) integer subject indices
            drop_repr: if True, zero out representation (for CFG)

        Returns:
            v_pred: (B, n_voxels) predicted velocity field
        """
        c, context = self._encode_and_condition(
            t, features_dict, drop_repr=drop_repr)
        return self._velocity_head(x_t, c, context, subject_id)

    def forward_with_dgs(self, t, x_t, features_dict, subject_id, drop_mask):
        """
        Forward pass with per-sample DGS (Dynamic Guidance Switching).

        Args:
            t: (B,) timestep values
            x_t: (B, n_voxels)
            features_dict: multimodal features
            subject_id: (B,) integer subject indices
            drop_mask: (B,) bool tensor — True to zero representation

        Returns:
            v_pred: (B, n_voxels)
        """
        c, context = self._encode_and_condition(
            t, features_dict, drop_mask=drop_mask)
        return self._velocity_head(x_t, c, context, subject_id)

    def forward_with_cfg(self, t, x_t, features_dict, subject_id,
                         cfg_scale=1.0):
        """Classifier-free guidance inference."""
        if cfg_scale == 1.0:
            return self.forward(t, x_t, features_dict, subject_id)

        v_cond = self.forward(
            t, x_t, features_dict, subject_id, drop_repr=False)
        v_uncond = self.forward(
            t, x_t, features_dict, subject_id, drop_repr=True)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_encoder_params(self):
        """Return encoder parameters (for separate LR)."""
        return list(self.encoder.parameters())

    def get_velocity_params(self):
        """Return velocity network parameters (everything except encoder)."""
        encoder_ids = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in encoder_ids]

    def param_count(self):
        """Return parameter counts for logging."""
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc_p = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad)
        block_p = sum(
            p.numel() for p in self.blocks.parameters() if p.requires_grad)
        cond_p = sum(
            p.numel() for p in self.cond_mlp.parameters() if p.requires_grad)
        ctx_p = 0
        if self.use_cross_attention:
            ctx_p = sum(
                p.numel() for p in self.context_proj.parameters()
                if p.requires_grad)
            ctx_p += sum(
                p.numel() for p in self.context_norm.parameters()
                if p.requires_grad)
        embed_p = sum(
            p.numel() for p in self.patch_embed.parameters()
            if p.requires_grad)
        out_p = sum(
            p.numel() for p in self.output_proj.parameters()
            if p.requires_grad)
        return {
            'encoder_M': enc_p / 1e6,
            'cond_M': cond_p / 1e6,
            'ctx_proj_M': ctx_p / 1e6,
            'blocks_M': block_p / 1e6,
            'embed_M': embed_p / 1e6,
            'output_M': out_p / 1e6,
            'total_M': total_p / 1e6,
        }

    def freeze_encoder(self):
        """Freeze the representation encoder."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the representation encoder."""
        for p in self.encoder.parameters():
            p.requires_grad = True
