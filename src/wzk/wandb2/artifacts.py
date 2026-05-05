from __future__ import annotations

from typing import Any

import wandb

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def log_artifact(name: str, artifact_type: str, paths: str | list[str], metadata: dict[str, Any] | None = None) -> None:
    """Create and log a W&B artifact from file(s).

    Args:
        name: Artifact name (e.g. "trained-model", "dataset-v1").
        artifact_type: Artifact type (e.g. "model", "dataset", "result").
        paths: File or directory path(s) to include.
        metadata: Optional metadata dict.
    """
    assert wandb.run is not None, "No active W&B run."
    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)

    if isinstance(paths, str):
        paths = [paths]

    for p in paths:
        artifact.add_file(p)

    wandb.log_artifact(artifact)
    logger.info(f"Logged artifact '{name}' ({artifact_type})")


def log_numpy(name: str, array_dict: dict[str, Any], artifact_type: str = "result") -> None:
    """Save numpy arrays as .npz and log as artifact.

    Args:
        name: Artifact name.
        array_dict: Dict of {key: np.ndarray} to save.
        artifact_type: Artifact type.
    """
    import tempfile

    import numpy as np

    assert wandb.run is not None, "No active W&B run."

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez_compressed(f.name, **array_dict)
        log_artifact(name=name, artifact_type=artifact_type, paths=f.name)
