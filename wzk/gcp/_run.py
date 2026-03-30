from __future__ import annotations

import atexit
import subprocess
import sys

from wzk.logger import setup_logger

from ._compute import create_instance, delete_instance, get_instance_status
from ._config import DEFAULT_PROJECT, GpuConfig, ProvisioningModel, VmConfig
from ._ssh import ssh_command, ssh_script, wait_for_ssh
from ._storage import gcs_cat, gcs_exists
from ._zones import create_with_zone_retry

logger = setup_logger(__name__)


def run_on_ephemeral_vm(
    vm_name: str,
    setup_script: str,
    run_script: str,
    *,
    gpu_type: str = "l4",
    project: str = DEFAULT_PROJECT,
    zone: str | None = None,
    image: str,
    gcs_prefix: str | None = None,
    completion_marker: str = "exitcode",
    delete_on_exit: bool = True,
    poll_interval: int = 30,
    push_branch: bool = True,
) -> int:
    """Run a workload on an ephemeral SPOT VM.

    Lifecycle:
    1. (Optional) Push current git branch
    2. Create SPOT VM (zone auto-retry or single zone)
    3. Wait for SSH
    4. Run setup_script via SSH (agent forwarding for git)
    5. Run run_script detached via nohup
    6. If gcs_prefix set: poll GCS for completion marker, fetch results
    7. Cleanup: delete VM on exit
    8. Return exit code
    """
    created_zone: str | None = None
    gpu = GpuConfig.from_gpu_type(gpu_type)

    config = VmConfig(
        name=vm_name,
        project=project,
        zone=zone or "us-central1-a",
        image=image,
        gpu=gpu,
        provisioning=ProvisioningModel.SPOT,
    )

    def _cleanup() -> None:
        if created_zone and delete_on_exit:
            logger.info("Deleting VM %s...", vm_name)
            try:
                delete_instance(project, created_zone, vm_name)
            except Exception:
                logger.warning("Failed to delete VM %s", vm_name)
        elif created_zone:
            logger.info("VM still running: %s (zone=%s)", vm_name, created_zone)

    atexit.register(_cleanup)

    # Step 1: Push current branch
    if push_branch:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        logger.info("Pushing branch '%s' to origin...", branch)
        subprocess.run(["git", "push", "-u", "origin", branch], check=True)

    # Step 2: Create VM
    logger.info("Creating GPU VM %s...", vm_name)
    if zone is not None:
        create_instance(config)
        created_zone = zone
    else:
        created_zone, gpu = create_with_zone_retry(config)

    logger.info("VM %s (%s, %s) in %s", vm_name, gpu.gpu_type, gpu.machine_type, created_zone)

    # Step 3: Wait for SSH
    logger.info("Waiting for SSH...")
    wait_for_ssh(vm_name, created_zone, project=project)

    # Step 4: Setup
    logger.info("Running setup script...")
    result = ssh_script(vm_name, created_zone, setup_script, project=project)
    if result.returncode != 0:
        logger.error("Setup failed:\n%s", result.stderr)
        return 1
    logger.info("Setup complete")

    # Step 5: Launch detached
    logger.info("Launching run script (detached)...")
    launch_cmd = f"cat > /tmp/run_payload.sh << 'PAYLOAD_EOF'\n{run_script}\nPAYLOAD_EOF\nchmod +x /tmp/run_payload.sh\nnohup /tmp/run_payload.sh > /dev/null 2>&1 &\necho \"Launched (PID $!)\""
    result = ssh_command(vm_name, created_zone, launch_cmd, project=project)
    logger.info(result.stdout.strip())

    # Step 6: Poll GCS
    if gcs_prefix is not None:
        marker_uri = f"{gcs_prefix}/{completion_marker}"
        logger.info("Polling for completion: %s", marker_uri)

        import time

        while True:
            time.sleep(poll_interval)

            status = get_instance_status(project, created_zone, vm_name)
            if status != "RUNNING":
                logger.error("VM was preempted or stopped (status: %s)", status)
                return 1

            if gcs_exists(marker_uri):
                logger.info("Workload finished.")
                break

            logger.info("Running...")

        exit_code_str = gcs_cat(marker_uri).strip()
        exit_code = int(exit_code_str) if exit_code_str.isdigit() else 1

        # Fetch log
        log_uri = f"{gcs_prefix}/output.log"
        if gcs_exists(log_uri):
            log_content = gcs_cat(log_uri)
            # Print last 30 lines
            lines = log_content.splitlines()
            for line in lines[-30:]:
                print(line)
            print(f"\nFull log: gsutil cat {log_uri}")

        return exit_code

    # No GCS polling (e.g., bench pushes to git)
    logger.info("Workload launched. No GCS polling configured.")
    logger.info(
        "Monitor: gcloud compute ssh %s --zone=%s --project=%s -- tail -f ~/bench.log", vm_name, created_zone, project
    )
    return 0


def _get_branch() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


if __name__ == "__main__":
    sys.exit(run_on_ephemeral_vm("test-vm", "echo setup", "echo run"))
