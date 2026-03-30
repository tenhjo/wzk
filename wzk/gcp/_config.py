from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum


class GpuType(StrEnum):
    L4 = "l4"
    T4 = "t4"


class ProvisioningModel(StrEnum):
    SPOT = "SPOT"
    STANDARD = "STANDARD"


@dataclass(frozen=True)
class GpuConfig:
    gpu_type: GpuType
    machine_type: str
    accelerator_type: str | None
    accelerator_count: int

    @classmethod
    def l4(cls) -> GpuConfig:
        return cls(gpu_type=GpuType.L4, machine_type="g2-standard-4", accelerator_type=None, accelerator_count=0)

    @classmethod
    def t4(cls) -> GpuConfig:
        return cls(
            gpu_type=GpuType.T4, machine_type="n1-standard-4", accelerator_type="nvidia-tesla-t4", accelerator_count=1
        )

    @classmethod
    def from_gpu_type(cls, gpu_type: str | GpuType) -> GpuConfig:
        gpu_type = GpuType(gpu_type)
        if gpu_type == GpuType.L4:
            return cls.l4()
        return cls.t4()


DEFAULT_PROJECT = os.environ["GCP_PROJECT"]
DEFAULT_ZONE = "us-central1-a"
DEFAULT_BOOT_DISK_SIZE_GB = 200
DEFAULT_BOOT_DISK_TYPE = "pd-ssd"
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

ROKIN_REPO = os.environ["ROKIN_REPO"]
ROBOT_ZOO_REPO = os.environ["ROBOT_ZOO_REPO"]
UV_CACHE_DIR = "/opt/uv-cache"


@dataclass
class VmConfig:
    name: str
    image: str
    project: str = DEFAULT_PROJECT
    zone: str = DEFAULT_ZONE
    boot_disk_size_gb: int = DEFAULT_BOOT_DISK_SIZE_GB
    boot_disk_type: str = DEFAULT_BOOT_DISK_TYPE
    gpu: GpuConfig = field(default_factory=GpuConfig.l4)
    provisioning: ProvisioningModel = ProvisioningModel.SPOT
    labels: dict[str, str] = field(default_factory=lambda: {"user": os.environ["GCP_USER_LABEL"]})
    scopes: list[str] = field(default_factory=lambda: list(DEFAULT_SCOPES))
