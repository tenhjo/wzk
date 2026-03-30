"""Bake a GCP GPU image with warm UV cache (Fire CLI).

Creates a temporary VM, runs uv sync + Eigen install, stops the VM,
replaces the image, and deletes the temporary VM.

Usage:
    python -m wzk.gcp.save_image bake
    python -m wzk.gcp.save_image bake --branch=main
"""

from __future__ import annotations

import fire

from wzk.logger import setup_logger

from ._compute import create_image, create_instance, delete_image, delete_instance, stop_instance
from ._config import (
    DEFAULT_PROJECT,
    DEFAULT_ZONE,
    ROBOT_ZOO_REPO,
    ROKIN_REPO,
    UV_CACHE_DIR,
    GpuConfig,
    ProvisioningModel,
    VmConfig,
)
from ._ssh import ssh_script, wait_for_ssh

logger = setup_logger(__name__)


def bake(
    *,
    image: str,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
    branch: str = "main",
    machine_type: str = "g2-standard-4",
) -> None:
    """Re-bake the GPU image with warm UV package cache."""
    from wzk.time2 import get_timestamp

    instance = f"bake-image-{get_timestamp()}"
    gpu = GpuConfig.l4()

    config = VmConfig(
        name=instance,
        project=project,
        zone=zone,
        image=image,
        gpu=GpuConfig(
            gpu_type=gpu.gpu_type,
            machine_type=machine_type,
            accelerator_type=gpu.accelerator_type,
            accelerator_count=gpu.accelerator_count,
        ),
        provisioning=ProvisioningModel.SPOT,
    )

    # Step 1: Create temporary VM
    logger.info("[1/6] Creating temporary VM: %s...", instance)
    create_instance(config)

    try:
        # Step 2: Wait for SSH
        logger.info("[2/6] Waiting for SSH...")
        wait_for_ssh(instance, zone, project=project)

        # Step 3: Warm up packages
        logger.info("[3/6] Warming up packages...")
        warmup_script = f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

sudo mkdir -p {UV_CACHE_DIR}
sudo chown $(whoami) {UV_CACHE_DIR}

mkdir -p ~/src
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
rm -rf ~/src/rokin ~/src/robot_zoo
git clone --depth 1 --branch {branch} {ROKIN_REPO} ~/src/rokin
git clone --depth 1 {ROBOT_ZOO_REPO} ~/src/robot_zoo

cd ~/src/rokin
UV_CACHE_DIR={UV_CACHE_DIR} uv sync --dev

UV_CACHE_DIR={UV_CACHE_DIR} uv run python -c 'import jax; assert jax.default_backend() == "gpu", f"Expected gpu, got {{jax.default_backend()}}"; print("JAX backend: gpu OK")'

if ! dpkg -s libeigen3-dev >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq libeigen3-dev
fi

rm -rf ~/src/rokin ~/src/robot_zoo

echo "Warm-up complete. UV cache size:"
du -sh {UV_CACHE_DIR}
"""
        result = ssh_script(instance, zone, warmup_script, project=project)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            logger.error("Warmup failed:\n%s", result.stderr)
            return

        # Step 4: Stop VM
        logger.info("[4/6] Stopping VM...")
        stop_instance(project, zone, instance)

        # Step 5: Replace image
        logger.info("[5/6] Replacing image '%s'...", image)
        delete_image(project, image)
        create_image(
            project,
            zone,
            image,
            source_disk=instance,
            description=f"Pre-warmed GPU image: CUDA 13, uv, Python 3.13, UV package cache at {UV_CACHE_DIR}",
        )

    finally:
        # Step 6: Delete temporary VM
        logger.info("[6/6] Deleting temporary VM...")
        try:
            delete_instance(project, zone, instance)
        except Exception:
            logger.warning("Failed to delete temporary VM %s", instance)

    logger.info("Done. Image '%s' updated.", image)


if __name__ == "__main__":
    fire.Fire({"bake": bake})
