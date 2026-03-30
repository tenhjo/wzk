from __future__ import annotations

import time

from google.cloud import storage

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse 'gs://bucket/path/to/blob' into (bucket_name, blob_name)."""
    if not gcs_uri.startswith("gs://"):
        msg = f"Invalid GCS URI: {gcs_uri}"
        raise ValueError(msg)
    parts = gcs_uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def gcs_upload(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to GCS."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("Uploaded %s -> %s", local_path, gcs_uri)


def gcs_download(gcs_uri: str, local_path: str) -> None:
    """Download a GCS blob to a local file."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logger.info("Downloaded %s -> %s", gcs_uri, local_path)


def gcs_cat(gcs_uri: str) -> str:
    """Read a GCS blob as a string."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def gcs_exists(gcs_uri: str) -> bool:
    """Check if a GCS blob exists."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()


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
