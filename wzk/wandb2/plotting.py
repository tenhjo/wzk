from __future__ import annotations

from typing import Any

import numpy as np
import wandb

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def log_figure(key: str, fig: Any, step: int | None = None, close: bool = True) -> None:
    """Log a matplotlib figure to W&B.

    Args:
        key: Metric name / panel title in W&B.
        fig: matplotlib Figure object.
        step: Optional global step.
        close: Close the figure after logging to free memory.
    """
    assert wandb.run is not None, "No active W&B run."
    import matplotlib.pyplot as plt

    wandb.log({key: wandb.Image(fig)}, step=step)
    if close:
        plt.close(fig)


def log_image(key: str, image: np.ndarray, step: int | None = None, caption: str | None = None) -> None:
    """Log a numpy image array (H, W) or (H, W, C) to W&B."""
    assert wandb.run is not None, "No active W&B run."
    wandb.log({key: wandb.Image(image, caption=caption)}, step=step)


def log_table(key: str, columns: list[str], data: list[list[Any]], step: int | None = None) -> None:
    """Log tabular data to W&B."""
    assert wandb.run is not None, "No active W&B run."
    table = wandb.Table(columns=columns, data=data)
    wandb.log({key: table}, step=step)


def log_histogram(key: str, values: np.ndarray, step: int | None = None, num_bins: int = 64) -> None:
    """Log a histogram to W&B."""
    assert wandb.run is not None, "No active W&B run."
    wandb.log({key: wandb.Histogram(values, num_bins=num_bins)}, step=step)


def log_path(key: str, path: np.ndarray, step: int | None = None, joint_names: list[str] | None = None) -> None:
    """Log a robot configuration path (n_wp, n_dof) as a W&B line plot per joint.

    Args:
        key: Base metric name.
        path: Array of shape (n_wp, n_dof).
        step: Optional global step.
        joint_names: Optional names for each DOF.
    """
    assert wandb.run is not None, "No active W&B run."
    n_wp, n_dof = path.shape
    if joint_names is None:
        joint_names = [f"j{i}" for i in range(n_dof)]

    data = {
        f"{key}/{name}": wandb.plot.line_series(
            xs=list(range(n_wp)),
            ys=[path[:, i].tolist()],
            keys=[name],
            title=f"{key}/{name}",
            xname="waypoint",
        )
        for i, name in enumerate(joint_names)
    }
    wandb.log(data, step=step)
