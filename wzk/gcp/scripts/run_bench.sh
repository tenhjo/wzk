#!/usr/bin/env bash
set -euo pipefail

# GPU benchmark on a temporary GCP spot VM.
#
# Creates a spot VM, clones the repo, runs all benchmarks, and pushes
# results back to the git remote — all inside the remote script so
# results survive SSH disconnection or spot preemption (as long as
# the benchmark section completed before termination).
#
# Usage:
#   ./tests/gcp/run_bench.sh
#   GPU_TYPE=t4 ZONE=us-east1-b ./tests/gcp/run_bench.sh

PROJECT="${PROJECT:-${GCP_PROJECT}}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-rokin-gpu-bench-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-${GCP_VM_IMAGE}}"
DELETE_INSTANCE="${DELETE_INSTANCE:-0}"

source "$(dirname "$0")/_gpu_config.sh"

ROKIN_REPO="${ROKIN_REPO}"
ROBOT_ZOO_REPO="${ROBOT_ZOO_REPO}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

SSH_FLAGS=(--ssh-flag="-A")

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed."
  exit 1
fi

if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q .; then
  echo "No active gcloud account. Run: gcloud auth login"
  exit 1
fi

if ! ssh-add -l &>/dev/null; then
  echo "No SSH key in agent. Run: ssh-add"
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

echo "[1/4] Pushing current branch '${BRANCH}' to origin..."
git push -u origin "${BRANCH}"

echo "[2/4] Creating GPU VM: ${INSTANCE} (${MACHINE_TYPE}, GPU_TYPE=${GPU_TYPE}, image=${IMAGE})..."
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

echo "[3/4] Waiting for SSH..."
wait_for_ssh

echo "[4/4] Uploading and launching benchmark on VM..."

# Build the remote script. Variable-expanded section first (branch/repo),
# then single-quoted heredoc for the rest (no local expansion).
REMOTE_SCRIPT="#!/usr/bin/env bash
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"

mkdir -p ~/src
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true

# Fresh clone
rm -rf ~/src/rokin ~/src/robot_zoo
git clone --branch ${BRANCH} ${ROKIN_REPO} ~/src/rokin
git clone --depth 1 ${ROBOT_ZOO_REPO} ~/src/robot_zoo
"

read -r -d '' REMOTE_SCRIPT_TAIL << 'SCRIPT_EOF' || true
cd ~/src/rokin

# Helper: commit and push any new results after each benchmark section.
push_results() {
  cd ~/src/rokin
  git add -f tests/benchmarks/results/ gpu_bench_*.txt 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "Benchmark results $(date -u +%Y-%m-%dT%H:%M:%S) $(hostname)"
    git push origin HEAD || echo "WARNING: git push failed"
  fi
}

# ── Setup ──────────────────────────────────────────────────────────
git config user.email "bench@rokin"
git config user.name "Benchmark Bot"

UV_CACHE_DIR=/opt/uv-cache uv sync --dev
if ! command -v nvcc >/dev/null 2>&1; then echo "nvcc missing: FFI CUDA build may be unavailable"; fi
nvidia-smi

# ── Internal: amortized throughput (PRIORITY) ─────────────────────
echo ""
echo "=== Internal benchmark: FK amortized throughput (CUDA) ==="
cd ~/src/rokin/tests/benchmarks
UV_CACHE_DIR=/opt/uv-cache uv run python bench_rokin_amortized.py \
  --batch-sizes 1,4,16,64,256,1024,4096,16384,32768 \
  --min-chain 32 --modes fwd,bwd \
  --output-dir ~/src/rokin/tests/benchmarks/results/gcp_l4 2>&1 | tee ~/src/rokin/gpu_bench_amortized.txt
cd ~/src/rokin

push_results

# ── cuRobo comparison ─────────────────────────────────────────────
echo ""
echo "=== External benchmark: rokin vs cuRobo ==="
UV_CACHE_DIR=/opt/uv-cache uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu130 2>/dev/null || true
UV_CACHE_DIR=/opt/uv-cache uv pip install --no-build-isolation \
  'nvidia-curobo @ git+https://github.com/NVlabs/curobo.git' 2>/dev/null || \
  echo "cuRobo installation failed — skipping cuRobo benchmark"
UV_CACHE_DIR=/opt/uv-cache uv run python tests/benchmarks/bench_curobo.py 2>&1 | tee gpu_bench_vs_curobo.txt || \
  echo "cuRobo benchmark failed (likely not installed)"

push_results

echo ""
echo "=== All benchmarks complete. Results pushed to git. ==="
SCRIPT_EOF

REMOTE_SCRIPT+="${REMOTE_SCRIPT_TAIL}"

# Write script to remote, then launch via nohup so it survives SSH disconnection.
echo "$REMOTE_SCRIPT" | gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
  "${SSH_FLAGS[@]}" -- "cat > /tmp/bench_remote.sh && chmod +x /tmp/bench_remote.sh"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" \
  "${SSH_FLAGS[@]}" -- "nohup bash /tmp/bench_remote.sh > ~/bench.log 2>&1 &"

echo ""
echo "Benchmark launched on ${INSTANCE} (detached via nohup)."
echo "Results will be pushed to git automatically when complete."
echo ""
echo "Monitor:  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT} -- tail -f ~/bench.log"
echo "Pull:     git pull"

echo ""
echo "Benchmarks finished. Pull results with: git pull"

if [[ "${DELETE_INSTANCE}" == "1" ]]; then
  echo "Deleting VM ${INSTANCE}..."
  gcloud compute instances delete "${INSTANCE}" \
    --zone="${ZONE}" --project="${PROJECT}" --quiet
fi
