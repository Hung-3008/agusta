"""BrainFlow — Seq2Seq Flow Matching with multitoken context fusion.

Seq2seq (``n_target_trs`` > 1, default):
    - MultiTokenFusion over modality-preserved context streams
    - RoPE temporal encoder over full context TRs (no causal mask)
    - Temporal slice to ``n_target_trs`` tokens aligned with fMRI targets
    - Decoder: DiT-X (cross-attention + AdaLN-Zero) or DiT-1D (additive + AdaLN-Zero)
    - Optional 7 Yeo network heads (``network_head``) with optional zero-init

Decoder types:
    ``decoder_type='ditx'`` (default): DiT-X from ManiFlow — 9-param AdaLN-Zero
        modulating Self-Attention + Cross-Attention + FFN. Context is queried at every
        block via cross-attention with RoPE on Q/K for temporal alignment.
    ``decoder_type='dit1d'``: Legacy 1D-DiT — 6-param AdaLN-Zero on Self-Attention + FFN.
        Context is added once to x_t_emb before entering the block stack.
    ``use_dit_decoder: false``: FiLM + cross-attention (oldest legacy path).

Ablation mode — ``use_regression=True``:
    Same DiT-X architecture, deterministic MSE regression (Table 3 ablation).
    The backbone is called at t=0 with x=zeros; no ODE integration at inference.

Usage:
        python src/train_brainflow.py --config src/configs/brainflow.yaml
        python src/train_brainflow_regression.py --config src/configs/brainflow_ablation_regression_ditx.yaml
"""

import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from .utils import recover_velocity_indi, IndiVelocityCallable, flow_train_time_sample, info_nce_loss
from .time_warp import TimeWarpNet, tensor_warp_schedule
from .hrf_source import AECNN_HRF_Source
from .velocity_net import VelocityNet

class BrainFlow(nn.Module):
    """Flow matching with auxiliary regression, optional contrastive loss, and CFG.

    Context encoding is handled inside ``VelocityNet`` (multitoken fusion,
    temporal encoder, optional slice to ``n_target_trs`` for seq2seq DiT).
    Regression pools detached context over time (aligned slice when using DiT).
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,

        tensor_fm_params: dict = None,
        indi_flow_matching: bool = False,
        indi_train_time_sqrt: bool = False,
        indi_min_denom: float = 1e-3,
        use_csfm: bool = False,
        csfm_var_reg_weight: float = 0.1,
        csfm_pcc_weight: float = 1.0,
        flow_loss_weight: float = 1.0,
        # --- Regression ablation (Table 3) ---
        use_regression: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.indi_flow_matching = indi_flow_matching
        self.indi_train_time_sqrt = indi_train_time_sqrt
        self.indi_min_denom = indi_min_denom
        self.use_csfm = use_csfm
        self.csfm_var_reg_weight = csfm_var_reg_weight
        self.csfm_pcc_weight = csfm_pcc_weight
        self.flow_loss_weight = flow_loss_weight
        self.use_regression = use_regression
        if use_regression and use_csfm:
            warnings.warn(
                "use_regression=True overrides use_csfm (SISFM is not used in regression mode).",
                stacklevel=2,
            )
            self.use_csfm = False

        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = VelocityNet(**vn_cfg)

        hidden_dim = vn_cfg.get("hidden_dim", 1024)
        use_subject_head = vn_cfg.get("use_subject_head", True)
        latent_dim = vn_cfg.get("latent_dim", hidden_dim)

        # --- Tensor Flow Matching (optional) ---
        self.use_tensor_fm = tensor_fm_params is not None
        if indi_flow_matching and self.use_tensor_fm:
            warnings.warn(
                "indi_flow_matching is ignored when tensor_fm is enabled "
                "(InDI recovery is only defined for linear OT paths).",
                stacklevel=2,
            )
        self._indi_effective = indi_flow_matching and not self.use_tensor_fm

        if self.use_tensor_fm:
            tfm = dict(tensor_fm_params)
            self.gamma_reg_weight = tfm.pop("gamma_reg_weight", 0.01)
            self.time_warp_net = TimeWarpNet(
                context_dim=hidden_dim,
                output_dim=output_dim,
                **tfm,
            )
        else:
            self.gamma_reg_weight = 0.0

        if self.use_csfm:
            self.hrf_source = AECNN_HRF_Source(
                context_dim=hidden_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                hrf_kernel_size=vn_cfg.get("hrf_kernel_size", 12)
            )

        # Log parameters
        vn_params = sum(p.numel() for p in self.velocity_net.parameters())
        csfm_params = sum(p.numel() for p in self.hrf_source.parameters()) if self.use_csfm else 0
        warp_params = sum(p.numel() for p in self.time_warp_net.parameters()) if self.use_tensor_fm else 0
        
        logger.info("VelocityNet: %s params", f"{vn_params:,}")
        if self.use_csfm:
            logger.info("  + HRF Source: %s params", f"{csfm_params:,}")
        logger.info(
            "Total params: %s (csfm=%s, tensor_fm=%s, regression=%s, modality_dims=%s)",
            f"{vn_params + csfm_params:,}",
            self.use_csfm,
            self.use_tensor_fm,
            self.use_regression,
            vn_cfg.get('modality_dims'),
        )

    def compute_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
        starting_distribution: torch.Tensor = None,
        skip_aux: bool = False,
        pre_encoded_context: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute flow + regression + contrastive loss.

        NSD-style gradient isolation:
          - Flow branch: gradient flows through encoder (updates fusion + DiT)
          - Reg branch: .detach() prevents conflict (updates reg head + proj only)

        Args:
            context: (B, T_ctx, total_dim) concatenated multimodal context.
            target:  (B, output_dim) or (B, n_target_trs, output_dim) ground-truth fMRI.
            subject_ids: (B,) long tensor of subject indices.
            skip_aux: if True, skip regression and contrastive losses (used when
                      context is zeroed out for CFG unconditional training).
            pre_encoded_context: optional pre-computed context encoding (e.g. when
                      the encoder is frozen and run under torch.no_grad()).

        Returns:
            dict with keys: total_loss, flow_loss, align_loss, cont_loss, gamma_reg.
        """
        # ------------------------------------------------------------------ #
        # Regression mode (Table 3 ablation: Regression DiT-X)               #
        # Same DiT-X backbone; no flow matching, no SISFM; pure MSE loss.    #
        # x is set to zeros and t=0, so the backbone acts as a deterministic #
        # encoder-decoder mapping context → fMRI.                            #
        # ------------------------------------------------------------------ #
        if self.use_regression:
            if pre_encoded_context is not None:
                context_encoded = pre_encoded_context
            else:
                context_encoded = self.velocity_net.encode_context_from_cond(context)

            B = target.shape[0]
            x_zeros = torch.zeros_like(target)           # no noise input
            t_zeros = torch.zeros(B, device=target.device, dtype=target.dtype)

            pred = self.velocity_net(
                x=x_zeros,
                t=t_zeros,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            mse_loss = F.mse_loss(pred, target)
            _zero = torch.tensor(0.0, device=target.device)
            return {
                "total_loss": mse_loss,
                "flow_loss":  mse_loss,    # alias so training loop logs are consistent
                "pcc_loss":   _zero,
                "var_reg_loss": _zero,
                "gamma_reg": _zero,
            }
        # 1. Encode context once (shared)
        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        else:
            context_encoded = self.velocity_net.encode_context_from_cond(context)

        # 2. Regression branch with gradient isolation (NSD improvement)
        # .detach() prevents regression from pulling the shared encoder
        _zero = torch.tensor(0.0, device=target.device)
        csfm_var_reg_loss = _zero
        csfm_pcc_loss = _zero
        x_0_csfm = None

        if skip_aux:
            reg_loss = _zero
            cont_loss = _zero
        else:
            if self.use_csfm:
                ctx_detached = context_encoded.detach()  # ⛔ no gradient to fusion
                ctx_transposed = ctx_detached.transpose(1, 2)
                ctx_pooled_reg = ctx_detached.mean(dim=1)
                
                # When encoder is frozen (pre_encoded_context provided), run the
                # entire CSFM branch under no_grad.  HRF source is frozen so its
                # output is fixed; allowing gradients through subject_layers here
                # would create conflicting signals with the flow loss path.
                _csfm_no_grad = pre_encoded_context is not None
                with torch.set_grad_enabled(not _csfm_no_grad):
                    mu_phi_latent, sigma_phi = self.hrf_source(ctx_transposed, ctx_pooled_reg)
                    
                    if self.velocity_net.use_subject_head:
                        if subject_ids is None:
                            subject_ids = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
                        fmri_pred = self.velocity_net.subject_layers(mu_phi_latent, subject_ids)
                    else:
                        fmri_pred = mu_phi_latent

                    epsilon = torch.randn_like(target)
                    if target.dim() == 2:
                        fmri_pred_compat = fmri_pred.squeeze(1)
                        sigma_phi_compat = sigma_phi.squeeze(1)
                    else:
                        fmri_pred_compat = fmri_pred
                        sigma_phi_compat = sigma_phi
                    x_0_csfm = fmri_pred_compat + sigma_phi_compat * epsilon
                    
                    # CSFM Losses (monitoring-only when encoder is frozen)
                    csfm_var_reg_loss = torch.mean(sigma_phi**2 - torch.log(sigma_phi**2 + 1e-8) - 1.0)
                    
                    # PCC loss between the clean base distribution (mu_phi) and target
                    _pred_flat = fmri_pred.flatten(1)  # (B, T*V)
                    _tgt_flat = target.flatten(1)
                    _pred_c = _pred_flat - _pred_flat.mean(dim=1, keepdim=True)
                    _tgt_c = _tgt_flat - _tgt_flat.mean(dim=1, keepdim=True)
                    _cov = (_pred_c * _tgt_c).sum(dim=1)
                    _std = torch.sqrt((_pred_c ** 2).sum(dim=1) * (_tgt_c ** 2).sum(dim=1) + 1e-8)
                    csfm_pcc_loss = (1.0 - _cov / _std).mean()

                if _csfm_no_grad:
                    x_0_csfm = x_0_csfm.detach()  # clean break for flow path

        # 4. Flow matching source distribution (x_0)
        x_1 = target
        if self.use_csfm and not skip_aux:
            x_0 = x_0_csfm
        else:
            x_0 = torch.randn_like(x_1)

        # 5. Flow matching (gradient flows to encoder)
        gamma_reg = _zero

        if self.use_tensor_fm:
            t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
            # Use NON-detached pooling so TimeWarpNet gradients flow through encoder
            ctx_pooled_flow = context_encoded.mean(dim=1)  # ✅ gradient flows
            gamma = self.time_warp_net(ctx_pooled_flow)     # (B, D)
            lambda_t, d_lambda_t = tensor_warp_schedule(gamma, t)

            x_t = lambda_t * x_1 + (1.0 - lambda_t) * x_0
            target_velocity = d_lambda_t * (x_1 - x_0)

            v_pred = self.velocity_net(
                x=x_t, t=t,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            flow_loss = F.mse_loss(v_pred, target_velocity)

            gamma_reg = gamma.pow(2).mean()
        else:
            t = flow_train_time_sample(
                x_1.shape[0],
                x_1.device,
                x_1.dtype,
                sqrt_bias_end=self.indi_train_time_sqrt and self._indi_effective,
            )
            # Manual OT interpolation — supports both (B, V) and (B, T, V) shapes
            t_bc = t
            while t_bc.dim() < x_1.dim():
                t_bc = t_bc.unsqueeze(-1)          # (B, 1, 1) for seq mode
            x_t = t_bc * x_1 + (1.0 - t_bc) * x_0

            if self._indi_effective:
                target_velocity = x_1 - x_t       # InDI: residual = (1-t)*(x_1-x_0)
            else:
                target_velocity = x_1 - x_0       # OT-CFM: constant velocity

            v_pred = self.velocity_net(
                x=x_t,
                t=t,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            flow_loss = F.mse_loss(v_pred, target_velocity)

        # 6. Total loss
        total_loss = self.flow_loss_weight * flow_loss
        if self.use_csfm:
            total_loss = total_loss + self.csfm_var_reg_weight * csfm_var_reg_loss
            total_loss = total_loss + self.csfm_pcc_weight * csfm_pcc_loss
            
        total_loss = total_loss + self.gamma_reg_weight * gamma_reg

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "pcc_loss": csfm_pcc_loss if self.use_csfm else _zero,
            "var_reg_loss": csfm_var_reg_loss if self.use_csfm else _zero,
            "gamma_reg": gamma_reg,
        }

    def _build_time_grid(
        self,
        n_timesteps: int,
        device: torch.device,
        dtype: torch.dtype,
        time_grid_warp: str | None,
        max_t: float = 1.0,
    ) -> torch.Tensor:
        """Monotone grid in [0, 1]. ``sqrt`` uses T = sqrt(s) for finer steps near t→1."""
        if n_timesteps < 2:
            n_timesteps = 2
        max_t = float(max(0.0, min(1.0, max_t)))
        s = torch.linspace(0, max_t, n_timesteps, device=device, dtype=dtype)
        if time_grid_warp is None or time_grid_warp in ("", "none", "linear"):
            return s
        if time_grid_warp == "sqrt":
            return torch.sqrt(torch.clamp(s / max(max_t, 1e-8), min=0.0, max=1.0)) * max_t
        raise ValueError(
            f"time_grid_warp must be None, 'linear', or 'sqrt', got {time_grid_warp!r}"
        )

    def _velocity_for_ode(self):
        """OT-CFM velocity for integration; InDI-trained nets need u/(1-t) recovery."""
        if self._indi_effective:
            return IndiVelocityCallable(self.velocity_net, self.indi_min_denom)
        return self.velocity_net

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
        cfg_scale: float = 0.0,
        temperature: float = 0.0,
        time_grid_warp: str | None = None,
        time_grid_max: float = 1.0,
        final_jump: bool = False,
        pre_encoded_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE (or single forward pass in regression mode).

        In ``use_regression`` mode the ODE is bypassed: the backbone is called
        once at t=0 with x=zeros, matching the training-time setup exactly.
        This ensures the ablation inference is identical to training.

        Supports single-TR and seq2seq modes.

        Single-step: returns (B, output_dim).
        Seq2seq:     returns (B, n_target_trs, output_dim).

        Args:
            context:               (B, T_ctx, total_dim) concatenated context.
                                   Can be None when pre_encoded_context is provided.
            n_timesteps:           Number of ODE steps.
            solver_method:         ``'midpoint'`` (default) or ``'euler'``.
            subject_ids:           (B,) subject indices.
            cfg_scale:             CFG guidance scale (0 = disabled).
            temperature:           Std-dev of starting noise (0 = zero init).
            time_grid_warp:        ``'sqrt'`` = finer steps near t→1.
            time_grid_max:         Upper integration bound in [0,1] to avoid t→1 singularity.
            final_jump:            If True and time_grid_max<1, perform a final residual jump.
            pre_encoded_context:   (B, n_target_trs, hidden_dim) pre-encoded context.
                                   When provided, skips encode_context_from_cond (saves compute
                                   when running the same context for multiple subjects).

        Returns:
            fmri_pred: (B, output_dim) or (B, n_target_trs, output_dim).
        """
        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
            B = context_encoded.shape[0]
            device = context_encoded.device
            dtype = context_encoded.dtype
        else:
            B = context.shape[0]
            device = context.device
            dtype = context.dtype
            context_encoded = self.velocity_net.encode_context_from_cond(context)
        n_target = getattr(self.velocity_net, 'n_target_trs', 1)

        # ------------------------------------------------------------------ #
        # Regression mode: single deterministic forward pass, no ODE.        #
        # ------------------------------------------------------------------ #
        if self.use_regression:
            x_zeros = torch.zeros(
                (B, n_target, self.output_dim) if n_target > 1 else (B, self.output_dim),
                device=device, dtype=dtype,
            )
            t_zeros = torch.zeros(B, device=device, dtype=dtype)
            return self.velocity_net(
                x=x_zeros,
                t=t_zeros,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )

        uncond_encoded = None
        if cfg_scale > 0:
            if context is not None:
                uncond_encoded = self.velocity_net.encode_context_from_cond(
                    torch.zeros_like(context)
                )
            else:
                # pre_encoded_context mode: create zero unconditional encoding directly
                uncond_encoded = torch.zeros_like(context_encoded)

        # --- Initialise x_0 ---
        if self.use_csfm:
            ctx_transposed = context_encoded.transpose(1, 2)
            ctx_pooled = context_encoded.mean(dim=1)
            mu_phi_latent, sigma_phi = self.hrf_source(ctx_transposed, ctx_pooled)
            
            if self.velocity_net.use_subject_head:
                if subject_ids is None:
                    subject_ids = torch.zeros(B, dtype=torch.long, device=device)
                mu_phi_fmri = self.velocity_net.subject_layers(mu_phi_latent, subject_ids)
            else:
                mu_phi_fmri = mu_phi_latent
            
            if n_target == 1:
                mu_phi_fmri_compat = mu_phi_fmri.squeeze(1)
                sigma_phi_compat = sigma_phi.squeeze(1)
            else:
                mu_phi_fmri_compat = mu_phi_fmri
                sigma_phi_compat = sigma_phi

            x = mu_phi_fmri_compat.to(device=device, dtype=dtype)
            if temperature > 0:
                epsilon = torch.randn_like(x)
                x = x + temperature * sigma_phi_compat * epsilon
        elif temperature > 0:
            shape = (B, n_target, self.output_dim) if n_target > 1 else (B, self.output_dim)
            x = temperature * torch.randn(*shape, device=device, dtype=dtype)
        else:
            shape = (B, n_target, self.output_dim) if n_target > 1 else (B, self.output_dim)
            x = torch.zeros(*shape, device=device, dtype=dtype)

        # --- Build time grid ---
        T_grid = self._build_time_grid(
            n_timesteps,
            device,
            dtype,
            time_grid_warp,
            max_t=time_grid_max,
        )

        # --- Velocity helper (handles InDI recovery) ---
        def _vel(x_in, t_scalar, ctx_enc):
            t_b = t_scalar.reshape(1).expand(B)
            u = self.velocity_net(
                x=x_in, t=t_b,
                pre_encoded_context=ctx_enc,
                subject_ids=subject_ids,
            )
            if self._indi_effective:
                u = recover_velocity_indi(u, t_scalar, self.indi_min_denom)
            return u

        # --- Manual ODE integration (works for any x shape) ---
        for i in range(len(T_grid) - 1):
            t_i = T_grid[i]
            dt = T_grid[i + 1] - T_grid[i]

            if solver_method == "midpoint":
                # Stage 1: velocity at t_i
                v1 = _vel(x, t_i, context_encoded)
                if cfg_scale > 0:
                    v1 = _vel(x, t_i, uncond_encoded) + cfg_scale * (v1 - _vel(x, t_i, uncond_encoded))
                # Stage 2: velocity at t_i + dt/2
                x_mid = x + 0.5 * dt * v1
                t_mid = t_i + 0.5 * dt
                v2 = _vel(x_mid, t_mid, context_encoded)
                if cfg_scale > 0:
                    v2 = _vel(x_mid, t_mid, uncond_encoded) + cfg_scale * (
                        v2 - _vel(x_mid, t_mid, uncond_encoded)
                    )
                x = x + dt * v2
            else:  # euler
                v = _vel(x, t_i, context_encoded)
                if cfg_scale > 0:
                    v = _vel(x, t_i, uncond_encoded) + cfg_scale * (v - _vel(x, t_i, uncond_encoded))
                x = x + dt * v

        # Optional final step for singularity-avoidance schedules that stop before t=1.
        if final_jump and float(T_grid[-1].item()) < 1.0:
            t_last = T_grid[-1]
            t_b = t_last.reshape(1).expand(B)
            u_last = self.velocity_net(
                x=x,
                t=t_b,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            if cfg_scale > 0:
                u_uncond = self.velocity_net(
                    x=x,
                    t=t_b,
                    pre_encoded_context=uncond_encoded,
                    subject_ids=subject_ids,
                )
                u_last = u_uncond + cfg_scale * (u_last - u_uncond)

            if self._indi_effective:
                # InDI predicts residual u ≈ x_1 - x_t at late t.
                x = x + u_last
            else:
                # Non-InDI models already predict velocity, so integrate remaining interval.
                x = x + (1.0 - t_last) * u_last

        return x
