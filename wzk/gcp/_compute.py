from __future__ import annotations

import os

from google.cloud import compute_v1

from wzk.logger import setup_logger

from ._config import ProvisioningModel, VmConfig

logger = setup_logger(__name__)


def _build_instance(config: VmConfig) -> compute_v1.Instance:
    """Build a compute_v1.Instance resource from VmConfig."""
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=f"projects/{config.project}/global/images/{config.image}",
            disk_size_gb=config.boot_disk_size_gb,
            disk_type=f"zones/{config.zone}/diskTypes/{config.boot_disk_type}",
        ),
    )

    network_interface = compute_v1.NetworkInterface(
        access_configs=[compute_v1.AccessConfig(name="External NAT", type="ONE_TO_ONE_NAT")],
    )

    scheduling = compute_v1.Scheduling(
        provisioning_model=config.provisioning.value,
        on_host_maintenance="TERMINATE",
    )
    if config.provisioning == ProvisioningModel.SPOT:
        scheduling.instance_termination_action = "STOP"

    service_account = compute_v1.ServiceAccount(scopes=config.scopes)

    metadata = compute_v1.Metadata(items=[compute_v1.Items(key="enable-oslogin", value="TRUE")])

    instance = compute_v1.Instance(
        name=config.name,
        machine_type=f"zones/{config.zone}/machineTypes/{config.gpu.machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        scheduling=scheduling,
        service_accounts=[service_account],
        labels=config.labels,
        metadata=metadata,
    )

    if config.gpu.accelerator_type is not None:
        instance.guest_accelerators = [
            compute_v1.AcceleratorConfig(
                accelerator_type=f"zones/{config.zone}/acceleratorTypes/{config.gpu.accelerator_type}",
                accelerator_count=config.gpu.accelerator_count,
            )
        ]

    return instance


def create_instance(config: VmConfig) -> None:
    """Create a VM instance. Blocks until the operation completes."""
    client = compute_v1.InstancesClient()
    instance = _build_instance(config)
    operation = client.insert(project=config.project, zone=config.zone, instance_resource=instance)
    operation.result()
    logger.info("Created instance %s in %s", config.name, config.zone)


def get_instance_status(project: str, zone: str, name: str) -> str | None:
    """Return the instance status string (RUNNING, TERMINATED, etc.) or None if not found."""
    client = compute_v1.InstancesClient()
    try:
        instance = client.get(project=project, zone=zone, instance=name)
        return instance.status
    except Exception:
        return None


def start_instance(project: str, zone: str, name: str) -> None:
    """Start a stopped instance."""
    client = compute_v1.InstancesClient()
    operation = client.start(project=project, zone=zone, instance=name)
    operation.result()
    logger.info("Started instance %s", name)


def stop_instance(project: str, zone: str, name: str) -> None:
    """Stop a running instance (preserves disk)."""
    client = compute_v1.InstancesClient()
    operation = client.stop(project=project, zone=zone, instance=name)
    operation.result()
    logger.info("Stopped instance %s", name)


def delete_instance(project: str, zone: str, name: str) -> None:
    """Delete an instance and its boot disk."""
    client = compute_v1.InstancesClient()
    operation = client.delete(project=project, zone=zone, instance=name)
    operation.result()
    logger.info("Deleted instance %s", name)


def delete_image(project: str, image: str) -> None:
    """Delete an image."""
    client = compute_v1.ImagesClient()
    try:
        operation = client.delete(project=project, image=image)
        operation.result()
        logger.info("Deleted image %s", image)
    except Exception:
        logger.warning("Image %s not found, skipping delete", image)


def create_image(project: str, zone: str, image: str, source_disk: str, description: str = "") -> None:
    """Create an image from a disk."""
    client = compute_v1.ImagesClient()
    image_resource = compute_v1.Image(
        name=image,
        source_disk=f"zones/{zone}/disks/{source_disk}",
        labels={"user": os.environ["GCP_USER_LABEL"]},
        description=description,
    )
    operation = client.insert(project=project, image_resource=image_resource)
    operation.result()
    logger.info("Created image %s from disk %s", image, source_disk)
