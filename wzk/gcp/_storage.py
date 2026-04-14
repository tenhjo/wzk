from __future__ import annotations

import subprocess
import time

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def gcs_upload(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to GCS."""
    subprocess.run(["gcloud", "storage", "cp", local_path, gcs_uri], check=True, capture_output=True, text=True)
    logger.info("Uploaded %s -> %s", local_path, gcs_uri)


def gcs_download(gcs_uri: str, local_path: str) -> None:
    """Download a GCS blob to a local file."""
    subprocess.run(["gcloud", "storage", "cp", gcs_uri, local_path], check=True, capture_output=True, text=True)
    logger.info("Downloaded %s -> %s", gcs_uri, local_path)


def gcs_cat(gcs_uri: str) -> str:
    """Read a GCS blob as a string."""
    result = subprocess.run(["gcloud", "storage", "cat", gcs_uri], capture_output=True, text=True, check=True)
    return result.stdout


def gcs_exists(gcs_uri: str) -> bool:
    """Check if a GCS blob exists."""
    result = subprocess.run(["gcloud", "storage", "ls", gcs_uri], capture_output=True, text=True, check=False)
    return result.returncode == 0


def gcs_poll(gcs_uri: str, *, interval: int = 30, timeout: int = 7200) -> None:
    """Poll until a GCS blob exists. Raises TimeoutError after timeout seconds."""
    elapsed = 0
    while elapsed < timeout:
        if gcs_exists(gcs_uri):
            logger.info("Found %s", gcs_uri)
            return
        logger.info("Polling %s (%ds elapsed)", gcs_uri, elapsed)
        time.sleep(interval)
        elapsed += interval
    msg = f"GCS poll timed out after {timeout}s for {gcs_uri}"
    raise TimeoutError(msg)
