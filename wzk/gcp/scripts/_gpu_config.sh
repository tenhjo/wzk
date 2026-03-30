#!/usr/bin/env bash
# Shared GPU VM configuration for all GCP scripts.
#
# GPU_TYPE selects the GPU and matching machine type:
#   l4  (default) — g2-standard-4 (L4 built-in, no --accelerator needed)
#   t4            — n1-standard-4 + --accelerator=type=nvidia-tesla-t4,count=1
#
# Usage (in other scripts):
#   source "$(dirname "$0")/_gpu_config.sh"
#   gcloud compute instances create ... "${GPU_CREATE_FLAGS[@]}" ...

GPU_TYPE="${GPU_TYPE:-l4}"

case "${GPU_TYPE}" in
  l4)
    MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
    GPU_CREATE_FLAGS=(--machine-type="${MACHINE_TYPE}")
    ;;
  t4)
    MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
    GPU_CREATE_FLAGS=(
      --machine-type="${MACHINE_TYPE}"
      --accelerator=type=nvidia-tesla-t4,count=1
    )
    ;;
  *)
    echo "Unknown GPU_TYPE='${GPU_TYPE}'. Use 'l4' or 't4'." >&2
    exit 1
    ;;
esac
