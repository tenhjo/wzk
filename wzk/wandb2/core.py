from __future__ import annotations

from typing import Any

import wandb
from wandb.sdk.wandb_run import Run

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def init(
    project: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    mode: str = "online",
    **kwargs: Any,
) -> Run:
    """Initialize a W&B run. Thin wrapper around wandb.init with sensible defaults."""
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        group=group,
        mode=mode,
        **kwargs,
    )
    logger.info(f"W&B run started: {run.name} ({run.url})")
    return run


def finish() -> None:
    """Finish the current W&B run."""
    if wandb.run is not None:
        wandb.finish()


def log(data: dict[str, Any], step: int | None = None, commit: bool = True) -> None:
    """Log a dictionary of scalars / metrics to the current W&B run."""
    assert wandb.run is not None, "No active W&B run. Call wandb2.init() first."
    wandb.log(data, step=step, commit=commit)


def log_config(config: dict[str, Any]) -> None:
    """Update the run config after init (e.g., when hyperparams are determined later)."""
    assert wandb.run is not None, "No active W&B run. Call wandb2.init() first."
    wandb.config.update(config, allow_val_change=True)
