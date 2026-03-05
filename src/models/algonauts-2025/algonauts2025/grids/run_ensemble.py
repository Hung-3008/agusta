# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from data_utils.infra import ConfDict
from modeling_utils.utils import run_grid

from ..main import Experiment  # type: ignore
from .defaults import PROJECT_NAME, SAVEDIR, default_config

GRID_NAME = "model_soup"

update = {
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "slurm_partition": "partition",
        "job_name": PROJECT_NAME,
    },
    "wandb_config.group": GRID_NAME,
    "save_checkpoints": False,
    "seed": None,
    "patience": None,
}

grid = {
    "data.layers": [[0, 0.5, 1], [0.5, 0.75, 1.0], [0.5, 1.], [0, 0.2, 0.4, 0.6, 0.8, 1.]],
    "loss.name": ["MSELoss", "PearsonLoss", "SmoothL1Loss", "HuberLoss"],
    "data.layer_aggregation": [None, "group_mean"],
    "brain_model_config.subject_embedding": [True, False],
    "brain_model_config.layer_aggregation": ["cat", "mean"],
    "brain_model_config.feature_aggregation": ["cat", "sum"],
    "brain_model_config.modality_dropout": [0.0, 0.2, 0.4],
}


if __name__ == "__main__":
    updated_config = ConfDict(default_config)
    updated_config.update(update)

    out = run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        job_name_keys=["wandb_config.name", "infra.job_name"],
        combinatorial=True,
        n_randomly_sampled=1000,
        overwrite=False,
        dry_run=False,
        infra_mode="force",
    )
