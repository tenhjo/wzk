from __future__ import annotations

import subprocess

from wzk.logger import setup_logger

from ._compute import create_instance
from ._config import GpuConfig, GpuType, VmConfig

logger = setup_logger(__name__)

L4_ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-c",
    "us-west1-a",
    "us-west1-b",
    "us-east4-a",
    "europe-west4-a",
]

T4_ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-east1-c",
    "us-west1-b",
    "us-east4-a",
    "europe-west4-a",
]


def create_with_zone_retry(config: VmConfig) -> tuple[str, GpuConfig]:
    """Try creating the VM across L4 zones, then fall back to T4 zones.

    Returns the zone where the VM was created and the GpuConfig used.
    Raises RuntimeError if all zones are exhausted.
    """
    for gpu_type, zones in [(GpuType.L4, L4_ZONES), (GpuType.T4, T4_ZONES)]:
        gpu = GpuConfig.from_gpu_type(gpu_type)
        if gpu_type == GpuType.T4:
            logger.info("L4 unavailable in all zones, falling back to T4...")

        for zone in zones:
            logger.info("Trying %s in %s...", gpu_type, zone)
            try:
                trial = VmConfig(
                    name=config.name,
                    project=config.project,
                    zone=zone,
                    image=config.image,
                    boot_disk_size_gb=config.boot_disk_size_gb,
                    boot_disk_type=config.boot_disk_type,
                    gpu=gpu,
                    provisioning=config.provisioning,
                    labels=config.labels,
                    scopes=config.scopes,
                )
                create_instance(trial)
                logger.info("Created %s (%s) in %s", config.name, gpu_type, zone)
                return zone, gpu
            except subprocess.CalledProcessError:
                logger.debug("Failed to create in %s/%s", gpu_type, zone)

    msg = f"Failed to create VM {config.name} in any zone. All GPUs stocked out."
    raise RuntimeError(msg)
