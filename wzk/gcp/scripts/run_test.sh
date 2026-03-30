#!/usr/bin/env bash
set -euo pipefail

# GPU smoke test on a temporary GCP spot VM.
#
# Resilient design:
#   - Auto-retries multiple zones with L4→T4 fallback on stockout
#   - Tests run detached (nohup) so SSH disconnects don't kill them
#   - VM uploads results to GCS — no SSH pipe dependency
#   - Polls GCS for completion, then prints results and deletes VM
#
# Usage:
#   ./tests/gcp/run_test.sh
#   GPU_TYPE=t4 ./tests/gcp/run_test.sh          # force T4
#   ZONE=us-east1-b ./tests/gcp/run_test.sh      # force zone (skip auto-retry)
#   DELETE_INSTANCE=0 ./tests/gcp/run_test.sh     # keep VM after tests

PROJECT="${PROJECT:-${GCP_PROJECT}}"
INSTANCE="${INSTANCE:-rokin-gpu-test-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-${GCP_VM_IMAGE}}"
DELETE_INSTANCE="${DELETE_INSTANCE:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

GCS_BASE="gs://${GCS_BUCKET}/rokin/gpu-tests"
GCS_PREFIX="${GCS_BASE}/${INSTANCE}"

ROKIN_REPO="${ROKIN_REPO}"
ROBOT_ZOO_REPO="${ROBOT_ZOO_REPO}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

SSH_FLAGS=(--ssh-flag="-A")

# Zones to try in order. Override with ZONE= to skip auto-retry.
L4_ZONES=(us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-west1-a us-west1-b us-east4-a europe-west4-a)
T4_ZONES=(us-central1-a us-central1-b us-east1-c us-west1-b us-east4-a europe-west4-a)

# ── Preflight checks ──────────────────────────────────────────────

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed." >&2; exit 1
fi
if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q .; then
  echo "No active gcloud account. Run: gcloud auth login" >&2; exit 1
fi
if ! ssh-add -l &>/dev/null; then
  echo "No SSH key in agent. Run: ssh-add" >&2; exit 1
fi

# ── Cleanup trap: always delete VM on exit ─────────────────────────

CREATED_ZONE=""
cleanup() {
  if [[ -n "${CREATED_ZONE}" && "${DELETE_INSTANCE}" == "1" ]]; then
    echo "Deleting VM ${INSTANCE}..."
    gcloud compute instances delete "${INSTANCE}" \
      --zone="${CREATED_ZONE}" --project="${PROJECT}" --quiet 2>/dev/null || true
  elif [[ -n "${CREATED_ZONE}" ]]; then
    echo "VM still running: ${INSTANCE} (zone=${CREATED_ZONE})"
    echo "  Delete: gcloud compute instances delete ${INSTANCE} --zone=${CREATED_ZONE} --project=${PROJECT} --quiet"
  fi
}
trap cleanup EXIT

# ── Helpers ────────────────────────────────────────────────────────

wait_for_ssh() {
  local zone="$1"
  for i in $(seq 1 30); do
    if gcloud compute ssh "${INSTANCE}" --zone="${zone}" --project="${PROJECT}" \
        --command="true" 2>/dev/null; then
      return 0
    fi
    echo "  Waiting for SSH... (attempt $i/30)"
    sleep 10
  done
  echo "SSH timed out." >&2; return 1
}

try_create_vm() {
  local gpu_type="$1" zone="$2"
  GPU_TYPE="${gpu_type}" source "$(dirname "$0")/_gpu_config.sh"

  gcloud compute instances create "${INSTANCE}" \
    --zone="${zone}" \
    --project="${PROJECT}" \
    "${GPU_CREATE_FLAGS[@]}" \
    --labels=user=${GCP_USER_LABEL} \
    --image="${IMAGE}" \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform 2>&1
}

run_ssh() {
  gcloud compute ssh "${INSTANCE}" --zone="${CREATED_ZONE}" --project="${PROJECT}" "$@"
}

# ── Step 1: Push ───────────────────────────────────────────────────

echo "[1/5] Pushing current branch '${BRANCH}' to origin..."
git push -u origin "${BRANCH}"

# ── Step 2: Create VM (with zone/GPU auto-retry) ──────────────────

echo "[2/5] Creating GPU VM..."

if [[ -n "${ZONE:-}" ]]; then
  GPU_TYPE="${GPU_TYPE:-l4}"
  echo "  Trying ${GPU_TYPE} in ${ZONE}..."
  if try_create_vm "${GPU_TYPE}" "${ZONE}" >/dev/null 2>&1; then
    CREATED_ZONE="${ZONE}"
  fi
else
  for gpu in l4 t4; do
    if [[ "${gpu}" == "l4" ]]; then
      zones=("${L4_ZONES[@]}")
    else
      echo "  L4 unavailable in all zones, falling back to T4..."
      zones=("${T4_ZONES[@]}")
    fi
    for zone in "${zones[@]}"; do
      echo "  Trying ${gpu} in ${zone}..."
      if try_create_vm "${gpu}" "${zone}" >/dev/null 2>&1; then
        CREATED_ZONE="${zone}"
        GPU_TYPE="${gpu}"
        break 2
      fi
    done
  done
fi

if [[ -z "${CREATED_ZONE}" ]]; then
  echo "Failed to create VM in any zone. All GPUs stocked out." >&2
  exit 1
fi

GPU_TYPE="${GPU_TYPE:-l4}" source "$(dirname "$0")/_gpu_config.sh"
echo "  VM: ${INSTANCE} (${GPU_TYPE}, ${MACHINE_TYPE}) in ${CREATED_ZONE}"

# ── Step 3: Wait for SSH ──────────────────────────────────────────

echo "[3/5] Waiting for SSH..."
wait_for_ssh "${CREATED_ZONE}"

# ── Step 4: Clone repos + launch tests (detached) ─────────────────

echo "[4/5] Setting up repos and launching tests..."
echo "  Results will be uploaded to: ${GCS_PREFIX}/"

# Phase A: Clone repos (needs SSH agent forwarding for git)
run_ssh "${SSH_FLAGS[@]}" -- bash << SETUP
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
mkdir -p ~/src
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
rm -rf ~/src/rokin ~/src/robot_zoo
git clone --depth 1 --branch ${BRANCH} ${ROKIN_REPO} ~/src/rokin
git clone --depth 1 ${ROBOT_ZOO_REPO} ~/src/robot_zoo
cd ~/src/rokin
UV_CACHE_DIR=/opt/uv-cache uv sync --dev
nvidia-smi
UV_CACHE_DIR=/opt/uv-cache uv run python -c 'import jax; print("JAX backend:", jax.default_backend())'
SETUP

# Phase B: Write test script and launch via nohup (survives SSH disconnect).
# The script uploads results to GCS when done — no SSH needed to retrieve them.
run_ssh --command="$(cat << 'LAUNCH'
cat > /tmp/run_tests.sh << 'TESTSCRIPT'
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/rokin

nvidia-smi >> ~/gpu_test.log 2>&1
echo "" >> ~/gpu_test.log

UV_CACHE_DIR=/opt/uv-cache uv run pytest -v --tb=long \
  tests/backends/cuda/test_cuda_ffi.py \
  tests/backends/cuda/test_cuda_codegen_targets.py \
  tests/backends/test_parity.py \
  tests/backends/test_jit_vmap.py \
  tests/backends/jax/test_fk.py \
  tests/backends/jax/test_hessian.py \
  tests/integration/test_metadata_parity.py \
  >> ~/gpu_test.log 2>&1
echo $? > ~/gpu_test.exitcode

# Upload results to GCS
gsutil cp ~/gpu_test.log "${GCS_PREFIX}/gpu_test.log"
gsutil cp ~/gpu_test.exitcode "${GCS_PREFIX}/gpu_test.exitcode"
TESTSCRIPT
chmod +x /tmp/run_tests.sh
LAUNCH
)" # Write the script

# Inject the GCS_PREFIX into the test script (it was inside single-quoted heredoc)
run_ssh --command="sed -i 's|\${GCS_PREFIX}|${GCS_PREFIX}|g' /tmp/run_tests.sh"

# Launch detached
run_ssh --command="nohup /tmp/run_tests.sh > /dev/null 2>&1 & echo 'Tests launched (PID '\$!')';"

# ── Step 5: Poll GCS for completion ───────────────────────────────

echo "[5/5] Polling for test completion (every ${POLL_INTERVAL}s)..."
echo "  Watching: ${GCS_PREFIX}/gpu_test.exitcode"

while true; do
  sleep "${POLL_INTERVAL}"

  # Check VM is still alive (spot preemption)
  vm_status=$(gcloud compute instances describe "${INSTANCE}" \
    --zone="${CREATED_ZONE}" --project="${PROJECT}" \
    --format="get(status)" 2>/dev/null || echo "DELETED")

  if [[ "${vm_status}" != "RUNNING" ]]; then
    echo "  VM was preempted or stopped (status: ${vm_status})."
    echo "  Tests did not complete. Re-run the script to retry."
    CREATED_ZONE=""  # prevent cleanup from trying to delete non-existent VM
    exit 1
  fi

  # Check GCS for completion marker
  if gsutil -q stat "${GCS_PREFIX}/gpu_test.exitcode" 2>/dev/null; then
    echo "  Tests finished! Results uploaded to GCS."
    break
  fi

  # Show progress via SSH (best-effort, non-fatal if SSH fails)
  progress=$(run_ssh --command="grep -c 'PASSED\|FAILED\|ERROR' ~/gpu_test.log 2>/dev/null" 2>/dev/null || echo "?")
  echo "  Running... (${progress} test results so far)"
done

# ── Fetch results from GCS ────────────────────────────────────────

EXIT_CODE=$(gsutil cat "${GCS_PREFIX}/gpu_test.exitcode" 2>/dev/null || echo "1")

echo ""
echo "===== TEST RESULTS ====="
gsutil cat "${GCS_PREFIX}/gpu_test.log" 2>/dev/null | tail -30
echo ""
echo "Full log: gsutil cat ${GCS_PREFIX}/gpu_test.log"

# VM cleanup happens via EXIT trap
exit "${EXIT_CODE}"
