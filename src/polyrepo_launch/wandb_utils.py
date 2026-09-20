"""
- description:
    Deprecate earlier W&B patches of a Polyrepo experiment cell when its
    replacement starts. Run names follow v{exp}.{minor}.{patch}_{cell}.
- usage:
    uv run python train.py
    # In train.py, after wandb.init(), call
    # polyrepo_launch.wandb_utils.deprecate_previous_patches(run).
    # The training environment must have wandb installed.
- user_story:
    content:
        A researcher starts a fresh patch of one experiment cell. Earlier
        patches receive the deprecated tag so W&B filters can hide them,
        while their existing tags and other experiment cells are preserved.
    was_generated_via_skill: false
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb


def add_short_names_to_config(current_run: wandb.Run) -> None:
    """Add every resolved launch knob under its short name to W&B config."""
    # Read the alias values captured by the launcher, including named groups.
    # Resolved aliases own these columns, including names such as lr that
    # the training config initially logs as a structured schedule.
    current_run.config.update(
        json.loads(os.environ["POLYREPO_SHORT_CONFIG"]), allow_val_change=True
    )


def deprecate_previous_patches(current_run: wandb.Run) -> None:
    """Tag existing lower patches of the initialized run's experiment cell."""
    import wandb

    version, cell = current_run.name.split("_", 1)
    exp, minor, patch = version.split(".")
    previous_names = [
        f"{exp}.{minor}.{previous}_{cell}"
        for previous in range(int(patch))
    ]
    if not previous_names:
        return

    # Query display names so skipped patches and W&B's separate run IDs
    # need no special handling. Exact names keep other cells untouched.
    runs = wandb.Api().runs(
        path=f"{current_run.entity}/{current_run.project}",
        filters={"displayName": {"$in": previous_names}},
    )
    for run in runs:
        if "deprecated" not in run.tags:
            run.tags = list(run.tags) + ["deprecated"]
            run.update()
