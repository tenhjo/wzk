"""GCP GPU image management (Fire CLI).

Usage:
    python -m wzk.gcp.save_image create_base --image=johten-gpu-cuda13
    python -m wzk.gcp.save_image create_base --image=johten-gpu-cuda13 --cuda_version=13-0
    python -m wzk.gcp.save_image bake --image=johten-gpu-cuda13
    python -m wzk.gcp.save_image bake --image=johten-gpu-cuda13 --branch=main
"""

from __future__ import annotations

import os
import subprocess
import time

import fire

from wzk.logger import setup_logger

from ._compute import create_image, create_instance, delete_image, delete_instance, stop_instance
from ._config import (
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_PROJECT,
    DEFAULT_ZONE,
    GpuConfig,
    ProvisioningModel,
    VmConfig,
)
from ._ssh import ssh_command, ssh_script, wait_for_ssh

ROKIN_REPO = os.environ["ROKIN_REPO"]
ROBOT_ZOO_REPO = os.environ["ROBOT_ZOO_REPO"]
UV_CACHE_DIR = "/opt/uv-cache"

logger = setup_logger(__name__)


def _gcp_name(prefix: str) -> str:
    """Generate a GCP-valid instance name (lowercase, digits, hyphens only)."""
    from wzk.time2 import get_timestamp

    return f"{prefix}-{get_timestamp().replace('_', '-').replace(':', '')}"


def _warmup_script(branch: str) -> str:
    """Script to warm UV cache, verify JAX GPU, and install Eigen."""
    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

sudo mkdir -p {UV_CACHE_DIR}
sudo chown -R $(whoami) {UV_CACHE_DIR}

mkdir -p ~/src ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
rm -rf ~/src/rokin ~/src/robot_zoo
git clone --depth 1 --branch {branch} {ROKIN_REPO} ~/src/rokin
git clone --depth 1 {ROBOT_ZOO_REPO} ~/src/robot_zoo

cd ~/src/rokin
UV_CACHE_DIR={UV_CACHE_DIR} uv sync --dev --group cuda

UV_CACHE_DIR={UV_CACHE_DIR} uv run python -c 'import jax; assert jax.default_backend() == "gpu", f"Expected gpu, got {{jax.default_backend()}}"; print("JAX backend: gpu OK")'

if ! dpkg -s libeigen3-dev >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq libeigen3-dev
fi

rm -rf ~/src/rokin ~/src/robot_zoo

echo "Warm-up complete. UV cache size:"
du -sh {UV_CACHE_DIR}
"""


def _wait_for_nvidia_driver(
    instance: str, zone: str, *, project: str = DEFAULT_PROJECT, max_attempts: int = 30, interval: int = 20
) -> None:
    """Poll until nvidia-smi succeeds."""
    for attempt in range(1, max_attempts + 1):
        result = ssh_command(instance, zone, "nvidia-smi", project=project)
        if result.returncode == 0:
            logger.info("NVIDIA driver ready (attempt %d/%d)", attempt, max_attempts)
            return
        logger.info("Waiting for NVIDIA driver... (attempt %d/%d)", attempt, max_attempts)
        time.sleep(interval)
    msg = f"NVIDIA driver not ready after {max_attempts * interval}s for {instance}"
    raise TimeoutError(msg)


BASE_IMAGE_FAMILY = "ubuntu-accelerator-2204-amd64-with-nvidia-580"
BASE_IMAGE_PROJECT = "ubuntu-os-accelerator-images"


def _try_create_base_vm(
    instance: str, project: str, zone: str, boot_disk_size_gb: int, provisioning: str = "SPOT"
) -> bool:
    """Try creating a VM in the given zone. Returns True on success."""
    args = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance,
        f"--project={project}",
        f"--zone={zone}",
        "--machine-type=g2-standard-4",
        f"--labels=user={os.environ['GCP_USER_LABEL']}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        f"--boot-disk-size={boot_disk_size_gb}GB",
        "--boot-disk-type=pd-ssd",
        f"--provisioning-model={provisioning}",
        "--maintenance-policy=TERMINATE",
        f"--image-family={BASE_IMAGE_FAMILY}",
        f"--image-project={BASE_IMAGE_PROJECT}",
    ]
    if provisioning == "SPOT":
        args.append("--instance-termination-action=STOP")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.returncode == 0


def create_base(
    *,
    image: str,
    cuda_version: str = "13-0",
    project: str = DEFAULT_PROJECT,
    zone: str | None = None,
    boot_disk_size_gb: int = DEFAULT_BOOT_DISK_SIZE_GB,
    provisioning: str = "SPOT",
    branch: str = "main",
) -> None:
    """Create a GPU base image from scratch.

    Starts from a Google Deep Learning VM (Ubuntu 22.04 + NVIDIA 580 driver),
    installs CUDA toolkit, uv, Eigen, and warms the UV package cache.
    Retries across L4 zones if no zone is specified.
    """
    from ._zones import L4_ZONES

    instance = _gcp_name("create-base")

    # Step 1: Create VM (with zone retry)
    logger.info("[1/7] Creating VM from %s: %s...", BASE_IMAGE_FAMILY, instance)
    zones = [zone] if zone else L4_ZONES
    created_zone: str | None = None
    for z in zones:
        logger.info("Trying zone %s...", z)
        if _try_create_base_vm(instance, project, z, boot_disk_size_gb, provisioning):
            created_zone = z
            break
        logger.debug("Zone %s unavailable", z)

    if created_zone is None:
        msg = "Failed to create VM in any zone. All L4 GPUs stocked out."
        raise RuntimeError(msg)

    logger.info("VM created in %s", created_zone)

    try:
        logger.info("[2/7] Waiting for SSH...")
        wait_for_ssh(instance, created_zone, project=project)
        _wait_for_nvidia_driver(instance, created_zone, project=project)

        logger.info("[3/7] Installing CUDA toolkit %s...", cuda_version)
        cuda_script = f"""\
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda/bin:$PATH"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -qq
sudo apt-get install -y -qq cuda-toolkit-{cuda_version}
rm cuda-keyring_1.1-1_all.deb
nvidia-smi
nvcc --version
"""
        result = ssh_script(instance, created_zone, cuda_script, project=project)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            logger.error("CUDA toolkit install failed:\n%s", result.stderr)
            return

        logger.info("[4/7] Installing uv...")
        uv_script = """\
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
"""
        result = ssh_script(instance, created_zone, uv_script, project=project)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            logger.error("uv install failed:\n%s", result.stderr)
            return

        logger.info("[5/7] Warming up packages...")
        result = ssh_script(instance, created_zone, _warmup_script(branch), project=project)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            logger.error("Warmup failed:\n%s", result.stderr)
            return

        logger.info("[6/7] Stopping VM...")
        stop_instance(project, created_zone, instance)

        logger.info("[7/7] Creating image '%s'...", image)
        description = f"Base GPU image: Ubuntu 22.04, NVIDIA 580, CUDA {cuda_version.replace('-', '.')}, uv, Python 3.13, UV cache at {UV_CACHE_DIR}"
        delete_image(project, image)
        create_image(project, created_zone, image, source_disk=instance, description=description)
    finally:
        logger.info("Deleting temporary VM...")
        try:
            delete_instance(project, created_zone, instance)
        except Exception:
            logger.warning("Failed to delete temporary VM %s", instance)

    logger.info("Done. Image '%s' created.", image)


def bake(
    *,
    image: str,
    project: str = DEFAULT_PROJECT,
    zone: str | None = None,
    branch: str = "main",
) -> None:
    """Re-bake the GPU image with warm UV package cache."""
    from ._zones import create_with_zone_retry

    instance = _gcp_name("bake-image")

    config = VmConfig(
        name=instance,
        project=project,
        zone=zone or DEFAULT_ZONE,
        image=image,
        gpu=GpuConfig.l4(),
        provisioning=ProvisioningModel.SPOT,
    )

    logger.info("[1/6] Creating temporary VM: %s...", instance)
    if zone is not None:
        create_instance(config)
        created_zone = zone
    else:
        created_zone, _gpu = create_with_zone_retry(config)

    try:
        logger.info("[2/6] Waiting for SSH...")
        wait_for_ssh(instance, created_zone, project=project)

        logger.info("[3/6] Warming up packages...")
        result = ssh_script(instance, created_zone, _warmup_script(branch), project=project)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            logger.error("Warmup failed:\n%s", result.stderr)
            return

        logger.info("[4/6] Stopping VM...")
        stop_instance(project, created_zone, instance)

        logger.info("[5/6] Replacing image '%s'...", image)
        delete_image(project, image)
        create_image(
            project,
            created_zone,
            image,
            source_disk=instance,
            description=f"Pre-warmed GPU image: CUDA 13, uv, Python 3.13, UV package cache at {UV_CACHE_DIR}",
        )
    finally:
        logger.info("[6/6] Deleting temporary VM...")
        try:
            delete_instance(project, created_zone, instance)
        except Exception:
            logger.warning("Failed to delete temporary VM %s", instance)

    logger.info("Done. Image '%s' updated.", image)


if __name__ == "__main__":
    fire.Fire({"create_base": create_base, "bake": bake})
