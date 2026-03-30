#!/usr/bin/env bash
set -euo pipefail

# Manage a persistent GCP GPU dev VM based on a pre-built image.
#
# The image (GCP_VM_IMAGE) has:
#   Ubuntu 22.04, NVIDIA L4 driver 580, CUDA 13.0, uv, Python 3.13,
#   GitHub SSH key.
#
# Repos are cloned to ~/src/ on first setup:
#   rokin, robot_zoo, ikx
#
# Usage:
#   ./tests/gcp/vm.sh start          # start (or create) the VM
#   ./tests/gcp/vm.sh stop           # stop the VM (preserves disk)
#   ./tests/gcp/vm.sh ssh            # SSH with agent forwarding
#   ./tests/gcp/vm.sh run <cmd>      # run a command via SSH
#   ./tests/gcp/vm.sh status         # show VM status
#   ./tests/gcp/vm.sh setup-repos    # clone repos to ~/src/
#   ./tests/gcp/vm.sh delete         # delete the VM entirely
#
# Environment overrides:
#   INSTANCE   VM name           (default: $GCP_VM_NAME)
#   ZONE       GCP zone          (default: us-central1-a)
#   PROJECT    GCP project       (default: $GCP_PROJECT)
#   IMAGE      base image name   (default: $GCP_VM_IMAGE)
#   GPU_TYPE   l4 (default) or t4

PROJECT="${PROJECT:-${GCP_PROJECT}}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-${GCP_VM_NAME}}"
IMAGE="${IMAGE:-${GCP_VM_IMAGE}}"

source "$(dirname "$0")/_gpu_config.sh"

SSH_FLAGS=(--ssh-flag="-A")

ROKIN_REPO="${ROKIN_REPO}"
ROBOT_ZOO_REPO="${ROBOT_ZOO_REPO}"

# ── Helpers ──────────────────────────────────────────────────────────

_check_gcloud() {
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "Error: gcloud CLI not installed." >&2; exit 1
  fi
}

_instance_exists() {
  gcloud compute instances describe "${INSTANCE}" \
    --zone="${ZONE}" --project="${PROJECT}" \
    --format="value(status)" 2>/dev/null
}

_wait_for_ssh() {
  echo "Waiting for SSH..."
  for i in $(seq 1 20); do
    if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
        --command="true" 2>/dev/null; then
      echo "SSH ready."
      return 0
    fi
    echo "  attempt $i/20"
    sleep 10
  done
  echo "SSH timed out." >&2; exit 1
}

_run_ssh() {
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
    "${SSH_FLAGS[@]}" -- "$@"
}

# ── Commands ─────────────────────────────────────────────────────────

cmd_start() {
  _check_gcloud
  local status
  status=$(_instance_exists)

  if [[ -z "${status}" ]]; then
    echo "Creating new VM '${INSTANCE}' (${MACHINE_TYPE}, GPU_TYPE=${GPU_TYPE}) from image '${IMAGE}'..."
    gcloud compute instances create "${INSTANCE}" \
      --zone="${ZONE}" \
      --project="${PROJECT}" \
      "${GPU_CREATE_FLAGS[@]}" \
      --labels=user=${GCP_USER_LABEL} \
      --image="${IMAGE}" \
      --boot-disk-size=200GB \
      --boot-disk-type=pd-ssd \
      --provisioning-model=SPOT \
      --instance-termination-action=STOP \
      --maintenance-policy=TERMINATE \
      --scopes=https://www.googleapis.com/auth/cloud-platform
  elif [[ "${status}" == "TERMINATED" || "${status}" == "STOPPED" ]]; then
    echo "Starting stopped VM '${INSTANCE}'..."
    gcloud compute instances start "${INSTANCE}" \
      --zone="${ZONE}" --project="${PROJECT}"
  elif [[ "${status}" == "RUNNING" ]]; then
    echo "VM '${INSTANCE}' is already running."
  else
    echo "VM '${INSTANCE}' is in state: ${status}"
    exit 1
  fi

  _wait_for_ssh

  echo ""
  echo "VM ready. Connect with:"
  echo "  ./tests/gcp/vm.sh ssh"
  echo "  ./tests/gcp/vm.sh run 'nvidia-smi'"
}

cmd_stop() {
  _check_gcloud
  echo "Stopping VM '${INSTANCE}'..."
  gcloud compute instances stop "${INSTANCE}" \
    --zone="${ZONE}" --project="${PROJECT}"
  echo "Stopped. Disk preserved — restart with: ./tests/gcp/vm.sh start"
}

cmd_ssh() {
  _check_gcloud
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
    "${SSH_FLAGS[@]}"
}

cmd_run() {
  _check_gcloud
  _run_ssh "$@"
}

cmd_status() {
  _check_gcloud
  local status
  status=$(_instance_exists)
  if [[ -z "${status}" ]]; then
    echo "VM '${INSTANCE}' does not exist."
  else
    echo "VM '${INSTANCE}': ${status}"
    if [[ "${status}" == "RUNNING" ]]; then
      gcloud compute instances describe "${INSTANCE}" \
        --zone="${ZONE}" --project="${PROJECT}" \
        --format="table(name,zone.basename(),machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
    fi
  fi
}

cmd_setup_repos() {
  _check_gcloud
  echo "Cloning repos to ~/src/ on '${INSTANCE}'..."
  _run_ssh bash -c '
    set -e
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    mkdir -p ~/src
    [ -d ~/src/rokin ] || git clone '"${ROKIN_REPO}"' ~/src/rokin
    [ -d ~/src/robot_zoo ] || git clone '"${ROBOT_ZOO_REPO}"' ~/src/robot_zoo
    echo "Repos ready in ~/src/"
    ls -1 ~/src/
  '
}

cmd_delete() {
  _check_gcloud
  echo "Deleting VM '${INSTANCE}' (this destroys the disk)..."
  gcloud compute instances delete "${INSTANCE}" \
    --zone="${ZONE}" --project="${PROJECT}" --quiet
  echo "Deleted."
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-help}" in
  start)        cmd_start ;;
  stop)         cmd_stop ;;
  ssh)          cmd_ssh ;;
  run)          shift; cmd_run "$@" ;;
  status)       cmd_status ;;
  setup-repos)  cmd_setup_repos ;;
  delete)       cmd_delete ;;
  *)
    echo "Usage: $0 {start|stop|ssh|run|status|setup-repos|delete}"
    echo ""
    echo "Manages a persistent GCP GPU dev VM (L4 + CUDA 13)."
    echo ""
    echo "  start         Start or create the VM"
    echo "  stop          Stop the VM (preserves disk)"
    echo "  ssh           SSH into the VM (with agent forwarding)"
    echo "  run <cmd>     Run a command on the VM"
    echo "  status        Show VM status"
    echo "  setup-repos   Clone rokin/robot_zoo/ikx to ~/src/"
    echo "  delete        Delete the VM and its disk"
    echo ""
    echo "Environment: INSTANCE=${INSTANCE} ZONE=${ZONE} PROJECT=${PROJECT}"
    exit 1
    ;;
esac
