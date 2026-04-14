"""Persistent GPU VM management (Fire CLI).

Usage:
    python -m wzk.gcp.vm start                                    # SPOT dev VM
    python -m wzk.gcp.vm start --name=johten-workstation --provisioning=STANDARD
    python -m wzk.gcp.vm ssh
    python -m wzk.gcp.vm ssh --name=johten-workstation
    python -m wzk.gcp.vm run 'nvidia-smi'
    python -m wzk.gcp.vm status
    python -m wzk.gcp.vm stop
    python -m wzk.gcp.vm delete
"""

from __future__ import annotations

import fire

from wzk.logger import setup_logger

from ._compute import create_instance, delete_instance, get_instance_status, start_instance, stop_instance
from ._config import (
    DEFAULT_PROJECT,
    DEFAULT_ZONE,
    GpuConfig,
    ProvisioningModel,
    VmConfig,
)
from ._ssh import ssh_command, ssh_interactive, wait_for_ssh

logger = setup_logger(__name__)


def start(
    name: str = "johten-gpu-dev",
    *,
    image: str,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
    gpu_type: str = "l4",
    provisioning: str = "SPOT",
) -> None:
    """Create or resume a persistent VM."""
    status = get_instance_status(project, zone, name)

    if status is None:
        gpu = GpuConfig.from_gpu_type(gpu_type)
        config = VmConfig(
            name=name,
            project=project,
            zone=zone,
            image=image,
            gpu=gpu,
            provisioning=ProvisioningModel(provisioning),
        )
        logger.info("Creating VM '%s' (%s, %s)...", name, gpu.machine_type, provisioning)
        create_instance(config)
    elif status in ("TERMINATED", "STOPPED"):
        logger.info("Starting stopped VM '%s'...", name)
        start_instance(project, zone, name)
    elif status == "RUNNING":
        logger.info("VM '%s' is already running.", name)
    else:
        logger.error("VM '%s' is in state: %s", name, status)
        return

    wait_for_ssh(name, zone, project=project)
    logger.info("VM ready. Connect with: python -m wzk.gcp.vm ssh --name=%s", name)


def stop(
    name: str = "johten-gpu-dev",
    *,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
) -> None:
    """Stop a VM (preserves disk)."""
    logger.info("Stopping VM '%s'...", name)
    stop_instance(project, zone, name)
    logger.info("Stopped. Restart with: python -m wzk.gcp.vm start --name=%s", name)


def ssh(
    name: str = "johten-gpu-dev",
    *,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
) -> None:
    """Open an interactive SSH session with agent forwarding."""
    ssh_interactive(name, zone, project=project)


def run(
    name: str = "johten-gpu-dev",
    *command: str,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
) -> None:
    """Run a command on the VM via SSH."""
    cmd = " ".join(command)
    result = ssh_command(name, zone, cmd, project=project)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")


def status(
    name: str = "johten-gpu-dev",
    *,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
) -> None:
    """Show VM status."""
    vm_status = get_instance_status(project, zone, name)
    if vm_status is None:
        logger.info("VM '%s' does not exist.", name)
    else:
        logger.info("VM '%s': %s", name, vm_status)


def delete(
    name: str = "johten-gpu-dev",
    *,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
) -> None:
    """Delete a VM and its disk."""
    logger.info("Deleting VM '%s'...", name)
    delete_instance(project, zone, name)
    logger.info("Deleted.")


if __name__ == "__main__":
    fire.Fire({
        "start": start,
        "stop": stop,
        "ssh": ssh,
        "run": run,
        "status": status,
        "delete": delete,
    })
