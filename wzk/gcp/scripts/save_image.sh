#!/usr/bin/env bash
set -euo pipefail

# Re-bake the GCP GPU image with a warm UV package cache.
#
# Creates a temporary VM from the current image, runs `uv sync --dev`
# to populate the UV cache at /opt/uv-cache, installs Eigen headers,
# then replaces the image.  Subsequent test/bench VMs skip all package
# downloads (~2 GB / ~40 s saved).
#
# Usage:
#   ./scripts/bake_gcp_gpu_image.sh

PROJECT="${PROJECT:-${GCP_PROJECT}}"
ZONE="${ZONE:-us-central1-a}"
IMAGE="${IMAGE:-${GCP_VM_IMAGE}}"
INSTANCE="bake-image-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"

ROKIN_REPO="${ROKIN_REPO}"
ROBOT_ZOO_REPO="${ROBOT_ZOO_REPO}"
BRANCH="${BRANCH:-main}"

SSH_FLAGS=(--ssh-flag="-A")
UV_CACHE="/opt/uv-cache"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed."
  exit 1
fi

wait_for_ssh() {
  for i in $(seq 1 20); do
    if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
        --command "true" 2>/dev/null; then
      return 0
    fi
    echo "  Waiting for SSH... (attempt $i/20)"
    sleep 10
  done
  echo "SSH timed out."
  exit 1
}

echo "[1/6] Creating temporary VM: ${INSTANCE}..."
gcloud compute instances create "${INSTANCE}" \
  --zone="${ZONE}" --project="${PROJECT}" \
  --machine-type="${MACHINE_TYPE}" --labels=user=${GCP_USER_LABEL} \
  --image="${IMAGE}" \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "[2/6] Waiting for SSH..."
wait_for_ssh

echo "[3/6] Warming up packages..."
REMOTE_SCRIPT="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"

# Persistent UV cache directory (survives image bake)
sudo mkdir -p ${UV_CACHE}
sudo chown \$(whoami) ${UV_CACHE}

# Clone repos
mkdir -p ~/src
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
rm -rf ~/src/rokin ~/src/robot_zoo
git clone --depth 1 --branch ${BRANCH} ${ROKIN_REPO} ~/src/rokin
git clone --depth 1 ${ROBOT_ZOO_REPO} ~/src/robot_zoo

# Install all Python packages (populates UV cache)
cd ~/src/rokin
UV_CACHE_DIR=${UV_CACHE} uv sync --dev

# Verify JAX sees the GPU
UV_CACHE_DIR=${UV_CACHE} uv run python -c 'import jax; assert jax.default_backend() == \"gpu\", f\"Expected gpu, got {jax.default_backend()}\"; print(\"JAX backend: gpu OK\")'

# Install Eigen headers (needed by C++ backend benchmarks)
if ! dpkg -s libeigen3-dev >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq libeigen3-dev
fi

# Clean up repos (they get freshly cloned each run anyway)
rm -rf ~/src/rokin ~/src/robot_zoo

echo 'Warm-up complete. UV cache size:'
du -sh ${UV_CACHE}
"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
  "${SSH_FLAGS[@]}" -- bash -c "${REMOTE_SCRIPT}"

echo "[4/6] Stopping VM..."
gcloud compute instances stop "${INSTANCE}" \
  --zone="${ZONE}" --project="${PROJECT}"

echo "[5/6] Replacing image '${IMAGE}'..."
# Delete existing image, then recreate from the baked disk
if gcloud compute images describe "${IMAGE}" --project="${PROJECT}" >/dev/null 2>&1; then
  gcloud compute images delete "${IMAGE}" --project="${PROJECT}" --quiet
fi

gcloud compute images create "${IMAGE}" \
  --project="${PROJECT}" \
  --source-disk="${INSTANCE}" \
  --source-disk-zone="${ZONE}" \
  --labels="user=${GCP_USER_LABEL}" \
  --description="Pre-warmed GPU image: CUDA 13, uv, Python 3.13, UV package cache at ${UV_CACHE}"

echo "[6/6] Deleting temporary VM..."
gcloud compute instances delete "${INSTANCE}" \
  --zone="${ZONE}" --project="${PROJECT}" --quiet

echo "Done. Image '${IMAGE}' updated."
