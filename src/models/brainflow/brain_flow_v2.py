"""
BrainFlow V2 — FlowFM-Style Direct Flow Matching for fMRI Generation.

Key changes from V1:
  - Trainable Representation Encoder f_φ (ViT-Small, DINOv2 init)
    takes raw images as input instead of pre-extracted features
  - FlowFM-style conditioning: concat(r, t) → MLP → adaLN-Zero
    (no prefix attention — conditions are injected via modulation only)
  - DGS operates on representation r, not on image tokens

Architecture:
  Image (B,3,224,224) → ViT Encoder → r (B, D_repr)
  Timestep t → sinusoidal_embed → t_emb (B, D)
  c = CondMLP(concat[proj(r), t_emb]) → (B, D)

  x_t (B, n_voxels) → patchify → patch_embed → (B, num_patches, D)
  → DiT Blocks × N (self-attn + FFN, adaLN-Zero from c)
  → output_proj → unpatchify → v_pred (B, n_voxels)
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Dict

import numpy as np
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


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BrainFlowV2Config:
    # fMRI dimensions
    n_voxels: int = 1000
    patch_size: int = 100       # 1000 → pad to 1000 → 10 patches

    # Transformer (velocity network)
    hidden_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # Multimodal Feature Encoder
    use_multimodal_features: bool = True
    feature_dims: Dict[str, int] = field(default_factory=dict)
    encoder_dim: int = 1024    # Output dimension of the multimodal encoder
    encoder_depth: int = 2     # Depth of the transformer encoder
    encoder_heads: int = 8

    # Conditioning
    cond_dim: int = 512        # internal conditioning dimension
    use_cross_attention: bool = True  # cross-attn with image patch tokens



# ─── Representation Encoder ──────────────────────────────────────────────────


class MultimodalFeatureEncoder(nn.Module):
    """
    Trainable Multimodal Feature Encoder f_φ.

    Input:  dict of features e.g. {"video": (B, T, D_v), "omni": (B, T, D_o)}
    Output: representation r (B, D_encoder) - cls token, and patch_tokens
    """

    def __init__(self, config: BrainFlowV2Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.encoder_dim

        # Projections for each modality to common embed_dim
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, self.embed_dim)
            for modality, dim in config.feature_dims.items()
        })

        # Cross-modal Transformer Encoder
        # Note: norm_first=True disables nested tensor optimization in PyTorch
        # (harmless, just slightly slower — suppress the noisy warning)
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
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_depth)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        # Modality embeddings
        self.modality_embeds = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
            for modality in config.feature_dims.keys()
        })

        # Final LayerNorm on representation (FlowFM-style)
        self.repr_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, features_dict):
        """
        features_dict: dict of tensors (B, T, D)
        Returns:
            cls_token: (B, embed_dim) — global representation
            patch_tokens: (B, N_patches, embed_dim) — spatial/temporal features
        """
        B = next(iter(features_dict.values())).shape[0]
        
        projected_seqs = []
        for modality, x in features_dict.items():
            if modality in self.projections:
                x_proj = self.projections[modality](x)
                x_proj = x_proj + self.modality_embeds[modality]
                projected_seqs.append(x_proj)
                
        if not projected_seqs:
            raise ValueError("No matching modalities found in features_dict")
            
        # Concatenate along time dimension
        seq = torch.cat(projected_seqs, dim=1) # (B, sum(T), embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        seq = torch.cat((cls_tokens, seq), dim=1) # (B, 1+sum(T), embed_dim)
        
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

        # Project representation to match hidden dim
        self.repr_proj = nn.Linear(encoder_dim, cond_dim)

        # Timestep embedding MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Merge MLP: concat(repr, t) → condition
        self.merge_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim),
        )

    def forward(self, r, t_emb):
        """
        r: (B, encoder_dim) — representation from ViT
        t_emb: (B, hidden_dim) — sinusoidal timestep embedding

        Returns: c (B, hidden_dim) — condition vector for adaLN
        """
        r_proj = self.repr_proj(r)        # (B, cond_dim)
        t_proj = self.t_mlp(t_emb)        # (B, cond_dim)
        merged = torch.cat([r_proj, t_proj], dim=-1)  # (B, cond_dim * 2)
        c = self.merge_mlp(merged)        # (B, hidden_dim)
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

        # Self-Attention (fMRI tokens only)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-Attention (fMRI tokens attend to image tokens)
        if use_cross_attention:
            self.cross_norm = nn.LayerNorm(
                hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # FFN (Pointwise Feedforward)
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
        # (shift1, scale1, gate1, [shift_ca, scale_ca, gate_ca,]
        #  shift2, scale2, gate2)
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
        c: (B, D) — condition vector from ConditioningMLP
        context: (B, M, D) — image patch tokens (for cross-attn)
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

        # 2) Cross-Attention with adaLN (fMRI Q, image K/V)
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


class BrainFlowV2(nn.Module):
    """
    BrainFlow V2 — FlowFM-style Direct Flow Matching for fMRI.

    Key differences from V1:
      - Trainable ViT encoder for image → representation
      - FlowFM-style conditioning: concat(r, t) → MLP → adaLN-Zero
      - No prefix attention (conditions via modulation only)
    """

    def __init__(self, config: Optional[BrainFlowV2Config] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BrainFlowV2Config(**kwargs)
        self.config = config
        D = config.hidden_dim

        # ── Representation Encoder f_φ ──
        self.encoder = MultimodalFeatureEncoder(config)

        # ── Conditioning MLP (uses CLS token for adaLN) ──
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

        self.voxel_perm = None
        self.voxel_inv_perm = None

        # Patch embedding: patch_size → D
        self.patch_embed = nn.Linear(config.patch_size, D)

        # ── Positional Embedding ──
        # Fully learnable positional embedding (index-based)
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
        self.output_proj = nn.Linear(D, config.patch_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for velocity network (not encoder)."""
        # Init velocity network layers
        for name, m in self.named_modules():
            if name.startswith('encoder'):
                continue  # skip encoder (pretrained)
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Re-zero output proj and adaLN
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

    def _patchify(self, x):
        """(B, n_voxels) → (B, num_patches, patch_size)"""
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        return x.view(x.shape[0], self.num_patches, self.config.patch_size)

    def _unpatchify(self, x):
        """(B, num_patches, patch_size) → (B, n_voxels)"""
        x = x.reshape(x.shape[0], -1)  # (B, padded_voxels)
        if self.pad_len > 0:
            x = x[:, :self.config.n_voxels]
        return x

    def forward(self, t, x_t, features_dict, drop_repr=False):
        """
        Predict velocity v(x_t, t | features_dict).

        Args:
            t: (B,) timestep values in [0, 1]
            x_t: (B, n_voxels) point on the probability path
            features_dict: dict of input features from SlidingWindowDataset
            drop_repr: if True, zero out representation (for DGS/CFG)

        Returns:
            v_pred: (B, n_voxels) predicted velocity field
        """
        B = x_t.shape[0]

        # ── Representation Encoder ──
        cls_token, patch_tokens = self.encoder(features_dict)
        # cls_token: (B, encoder_dim), patch_tokens: (B, N, encoder_dim)

        # DGS: zero-out representation
        if drop_repr:
            cls_token = torch.zeros_like(cls_token)
            patch_tokens = torch.zeros_like(patch_tokens)

        # ── Conditioning (adaLN via CLS token) ──
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)  # (B, D)
        c = self.cond_mlp(cls_token, t_emb)  # (B, D)

        # ── Context for cross-attention (patch tokens) ──
        context = None
        if self.use_cross_attention:
            context = self.context_norm(
                self.context_proj(patch_tokens))  # (B, N, D)

        # ── fMRI tokens ──
        x_patches = self._patchify(x_t)             # (B, num_patches, patch_size)
        x_tokens = self.patch_embed(x_patches)       # (B, num_patches, D)

        # Positional embedding
        x_tokens = x_tokens + self.patch_pos_embed
        # (B, num_patches, D)

        # ── DiT Blocks (self-attn + cross-attn + FFN) ──
        for block in self.blocks:
            x_tokens = block(x_tokens, c, context)

        # ── Output head ──
        mod_params = self.final_adaLN(c)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_tokens), shift, scale)
        x_out = self.output_proj(x_out)  # (B, 128, 124)

        # Unpatchify → (B, n_voxels)
        v_pred = self._unpatchify(x_out)
        return v_pred

    def forward_with_cfg(self, t, x_t, features_dict, cfg_scale=1.0):
        """
        Classifier-free guidance inference.
        Runs conditional + unconditional forward pass and interpolates.
        """
        if cfg_scale == 1.0:
            return self.forward(t, x_t, features_dict)

        # Conditional
        v_cond = self.forward(t, x_t, features_dict, drop_repr=False)
        # Unconditional
        v_uncond = self.forward(t, x_t, features_dict, drop_repr=True)

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
