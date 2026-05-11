from __future__ import annotations

import subprocess
import time

from wzk.logger import setup_logger

from ._config import DEFAULT_PROJECT

logger = setup_logger(__name__)


def _base_args(instance: str, zone: str, project: str) -> list[str]:
    return ["gcloud", "compute", "ssh", instance, f"--zone={zone}", f"--project={project}"]


def ssh_command(
    instance: str, zone: str, command: str, *, project: str = DEFAULT_PROJECT
) -> subprocess.CompletedProcess[str]:
    """Run a command on the instance via SSH and return the result."""
    args = [*_base_args(instance, zone, project), f"--command={command}"]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def ssh_interactive(instance: str, zone: str, *, project: str = DEFAULT_PROJECT) -> None:
    """Open an interactive SSH session with agent forwarding."""
    args = [*_base_args(instance, zone, project), "--ssh-flag=-A"]
    subprocess.run(args, check=False)


def ssh_script(
    instance: str, zone: str, script: str, *, project: str = DEFAULT_PROJECT
) -> subprocess.CompletedProcess[str]:
    """Run a multi-line shell script via SSH with agent forwarding (stdin heredoc)."""
    args = [*_base_args(instance, zone, project), "--ssh-flag=-A", "--", "bash"]
    return subprocess.run(args, input=script, capture_output=True, text=True, check=False)


def scp_to(
    local_path: str,
    instance: str,
    remote_path: str,
    zone: str,
    *,
    project: str = DEFAULT_PROJECT,
) -> subprocess.CompletedProcess[str]:
    """Copy a local file to the instance via gcloud compute scp."""
    args = [
        "gcloud",
        "compute",
        "scp",
        local_path,
        f"{instance}:{remote_path}",
        f"--zone={zone}",
        f"--project={project}",
    ]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def wait_for_ssh(
    instance: str, zone: str, *, project: str = DEFAULT_PROJECT, max_attempts: int = 30, interval: int = 10
) -> None:
    """Wait until SSH is reachable. Raises TimeoutError after max_attempts."""
    for attempt in range(1, max_attempts + 1):
        result = ssh_command(instance, zone, "true", project=project)
        if result.returncode == 0:
            logger.info("SSH ready on %s (attempt %d/%d)", instance, attempt, max_attempts)
            return
        logger.info("Waiting for SSH on %s (attempt %d/%d)", instance, attempt, max_attempts)
        time.sleep(interval)
    msg = f"SSH timed out after {max_attempts * interval}s for {instance}"
    raise TimeoutError(msg)
