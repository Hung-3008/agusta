"""ODE solver configuration dataclass."""

from dataclasses import dataclass

from src.utils.utils import InferenceStrategyConfig


@dataclass
class SolverConfig:
    """ODE solver settings, loaded from config and optionally overridden by CLI."""
    n_timesteps: int = 30
    method: str = "midpoint"
    temperature: float = 0.05
    cfg_scale: float = 0.0
    time_grid_warp: str | None = None
    time_grid_max: float = 1.0
    final_jump: bool = False
    use_pruned_sampling: bool = False
    prune_k: int = 5
    n_seeds: int = 1
    ensemble_mode: str = "none"
    base_seed: int = 1234

    @classmethod
    def from_cfg(cls, cfg: dict, args) -> "SolverConfig":
        s = cfg.get("solver_args", {})
        return cls(
            n_timesteps=args.n_timesteps or s.get("time_points", 30),
            method=args.solver_method or s.get("method", "midpoint"),
            temperature=args.temperature if args.temperature is not None else s.get("temperature", 0.05),
            cfg_scale=args.cfg_scale if args.cfg_scale is not None else s.get("cfg_scale", 0.0),
            time_grid_warp=args.time_grid_warp if args.time_grid_warp is not None else s.get("time_grid_warp", None),
            time_grid_max=args.time_grid_max if args.time_grid_max is not None else s.get("time_grid_max", 1.0),
            final_jump=args.final_jump if args.final_jump is not None else s.get("final_jump", False),
            use_pruned_sampling=args.use_pruned_sampling if args.use_pruned_sampling is not None else s.get("use_pruned_sampling", False),
            prune_k=args.prune_k if args.prune_k is not None else s.get("prune_k", 5),
            n_seeds=args.n_seeds if args.n_seeds is not None else s.get("n_seeds", 1),
            ensemble_mode=args.ensemble_mode if args.ensemble_mode is not None else s.get("ensemble_mode", "none"),
            base_seed=args.base_seed if args.base_seed is not None else s.get("base_seed", 1234),
        )

    def as_synth_kwargs(self) -> dict:
        kw = dict(
            n_timesteps=self.n_timesteps,
            solver_method=self.method,
            temperature=self.temperature,
            time_grid_max=self.time_grid_max,
            final_jump=self.final_jump,
        )
        if self.cfg_scale > 0:
            kw["cfg_scale"] = self.cfg_scale
        if self.time_grid_warp:
            kw["time_grid_warp"] = self.time_grid_warp
        return kw

    def as_strategy_config(self) -> InferenceStrategyConfig:
        return InferenceStrategyConfig(
            use_pruned_sampling=bool(self.use_pruned_sampling),
            prune_k=max(1, int(self.prune_k)),
            n_seeds=max(1, int(self.n_seeds)),
            ensemble_mode=(self.ensemble_mode or "none").lower(),
            base_seed=int(self.base_seed),
        )
