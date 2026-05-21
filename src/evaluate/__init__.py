"""BrainFlow evaluation package.

Provides modular evaluation and submission pipeline for BrainFlow models.
Split from the original monolithic evaluate_brainflow.py for maintainability.

Modules:
    config        — SolverConfig dataclass for ODE solver settings.
    model_runner  — ModelRunner wraps model build/load/inference.
    data_helpers  — Context/fMRI/window loading and PCC computation.
    submission    — Submission assembly, sample count loaders, denormalization.
    run_s6        — S6 validation with ground-truth PCC + extended metrics.
    run_s7        — S7 blind submission (sequential and parallel modes).
    run_ood       — OOD blind submission.
"""

from src.evaluate.config import SolverConfig
from src.evaluate.model_runner import ModelRunner
