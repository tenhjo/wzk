from __future__ import annotations

import json
import os
import subprocess

from wzk.logger import setup_logger

from ._config import ProvisioningModel, VmConfig

logger = setup_logger(__name__)


def _run_gcloud(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a gcloud command and return the result."""
    result = subprocess.run(["gcloud", *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        logger.error("gcloud %s failed:\n%s", args[1] if len(args) > 1 else args[0], result.stderr.strip())
        result.check_returncode()
    return result


def create_instance(config: VmConfig) -> None:
    """Create a VM instance. Blocks until the operation completes."""
    args = [
        "compute",
        "instances",
        "create",
        config.name,
        f"--project={config.project}",
        f"--zone={config.zone}",
        f"--machine-type={config.gpu.machine_type}",
        f"--image=projects/{config.project}/global/images/{config.image}",
        f"--boot-disk-size={config.boot_disk_size_gb}GB",
        f"--boot-disk-type={config.boot_disk_type}",
        f"--provisioning-model={config.provisioning.value}",
        "--maintenance-policy=TERMINATE",
        f"--scopes={','.join(config.scopes)}",
        "--metadata=enable-oslogin=TRUE",
    ]

    if config.labels:
        label_str = ",".join(f"{k}={v}" for k, v in config.labels.items())
        args.append(f"--labels={label_str}")

    if config.provisioning == ProvisioningModel.SPOT:
        args.append("--instance-termination-action=STOP")

    if config.gpu.accelerator_type is not None:
        args.append(f"--accelerator=type={config.gpu.accelerator_type},count={config.gpu.accelerator_count}")

    _run_gcloud(*args)
    logger.info("Created instance %s in %s", config.name, config.zone)


def get_instance_status(project: str, zone: str, name: str) -> str | None:
    """Return the instance status string (RUNNING, TERMINATED, etc.) or None if not found."""
    result = _run_gcloud(
        "compute",
        "instances",
        "describe",
        name,
        f"--project={project}",
        f"--zone={zone}",
        "--format=json",
        check=False,
    )
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    return data["status"]


def start_instance(project: str, zone: str, name: str) -> None:
    """Start a stopped instance."""
    _run_gcloud("compute", "instances", "start", name, f"--project={project}", f"--zone={zone}")
    logger.info("Started instance %s", name)


def stop_instance(project: str, zone: str, name: str) -> None:
    """Stop a running instance (preserves disk)."""
    _run_gcloud("compute", "instances", "stop", name, f"--project={project}", f"--zone={zone}")
    logger.info("Stopped instance %s", name)


def delete_instance(project: str, zone: str, name: str) -> None:
    """Delete an instance and its boot disk."""
    _run_gcloud("compute", "instances", "delete", name, f"--project={project}", f"--zone={zone}", "--quiet")
    logger.info("Deleted instance %s", name)


def delete_image(project: str, image: str) -> None:
    """Delete an image."""
    result = _run_gcloud("compute", "images", "delete", image, f"--project={project}", "--quiet", check=False)
    if result.returncode == 0:
        logger.info("Deleted image %s", image)
    else:
        logger.warning("Image %s not found, skipping delete", image)


def create_image(project: str, zone: str, image: str, source_disk: str, description: str = "") -> None:
    """Create an image from a disk."""
    args = [
        "compute",
        "images",
        "create",
        image,
        f"--project={project}",
        f"--source-disk={source_disk}",
        f"--source-disk-zone={zone}",
        f"--labels=user={os.environ['GCP_USER_LABEL']}",
    ]
    if description:
        args.append(f"--description={description}")
    _run_gcloud(*args)
    logger.info("Created image %s from disk %s", image, source_disk)
