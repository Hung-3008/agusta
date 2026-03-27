"""CSFM Transport — adapted from https://github.com/junwankimm/CSFM"""

from .transport import Transport, Sampler, ModelType, PathType, WeightType


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    time_dist_type="uniform",
    time_dist_shift=1.0,
):
    """Create Transport object for flow matching.

    Args:
        path_type: "Linear", "GVP", or "VP"
        prediction: "velocity", "noise", or "score"
        loss_weight: None, "velocity", or "likelihood"
        time_dist_type: "uniform" or "logit-normal_mu_sigma"
        time_dist_shift: shift for time distribution (≥ 1.0)
    """
    model_type = {
        "velocity": ModelType.VELOCITY,
        "noise": ModelType.NOISE,
        "score": ModelType.SCORE,
    }[prediction]

    loss_type = {
        "velocity": WeightType.VELOCITY,
        "likelihood": WeightType.LIKELIHOOD,
    }.get(loss_weight, WeightType.NONE)

    path_enum = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }[path_type]

    # Stable defaults for velocity + linear/GVP path
    if path_enum == PathType.VP:
        train_eps, sample_eps = 1e-5, 1e-3
    elif model_type != ModelType.VELOCITY:
        train_eps, sample_eps = 1e-3, 1e-3
    else:
        train_eps, sample_eps = 0, 0

    return Transport(
        model_type=model_type,
        path_type=path_enum,
        loss_type=loss_type,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
