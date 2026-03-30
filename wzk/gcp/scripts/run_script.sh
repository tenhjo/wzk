#!/usr/bin/env bash
set -euo pipefail

# Run a specific Python script on a temporary GCP GPU VM.
#
# Resilient design (same as run_test.sh):
#   - Auto-retries multiple zones with L4→T4 fallback on stockout
#   - Script runs detached (nohup) so SSH disconnects don't kill it
#   - VM uploads results to GCS — no SSH pipe dependency
#   - Polls GCS for completion, then prints results and deletes VM
#
# Usage:
#   ./tests/gcp/run_script.sh /tmp/bench_fk_ee_wjac.py
#   ./tests/gcp/run_script.sh scripts/bench/bench_rokin_amortized.py --batch-sizes 1,64,1024
#   GPU_TYPE=t4 ./tests/gcp/run_script.sh /tmp/my_script.py
#   ZONE=us-east1-b ./tests/gcp/run_script.sh /tmp/my_script.py   # force zone (skip auto-retry)
#   DELETE_INSTANCE=0 ./tests/gcp/run_script.sh /tmp/my_script.py  # keep VM after run

PROJECT="${PROJECT:-${GCP_PROJECT}}"
INSTANCE="${INSTANCE:-rokin-gpu-run-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-${GCP_VM_IMAGE}}"
DELETE_INSTANCE="${DELETE_INSTANCE:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

GCS_BASE="gs://${GCS_BUCKET}/rokin/gpu-scripts"
GCS_PREFIX="${GCS_BASE}/${INSTANCE}"

ROKIN_REPO="${ROKIN_REPO}"
ROBOT_ZOO_REPO="${ROBOT_ZOO_REPO}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

SSH_FLAGS=(--ssh-flag="-A")

SCRIPT_PATH="${1:?Usage: $0 <script.py> [args...]}"
shift
SCRIPT_ARGS="$*"

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

echo "[1/6] Pushing current branch '${BRANCH}' to origin..."
git push -u origin "${BRANCH}"

# ── Step 2: Create VM (with zone/GPU auto-retry) ──────────────────

echo "[2/6] Creating GPU VM..."

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

echo "[3/6] Waiting for SSH..."
wait_for_ssh "${CREATED_ZONE}"

# ── Step 4: Clone repos + install ─────────────────────────────────

echo "[4/6] Setting up repos..."

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

# ── Step 5: Upload script + launch detached ───────────────────────

echo "[5/6] Launching script (detached)..."
echo "  Results will be uploaded to: ${GCS_PREFIX}/"

# Upload script if outside repo
REMOTE_SCRIPT_PATH=""
REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ "${SCRIPT_PATH}" == /* ]] && [[ ! "${SCRIPT_PATH}" == "${REPO_ROOT}"/* ]]; then
  REMOTE_SCRIPT_PATH="/tmp/$(basename "${SCRIPT_PATH}")"
  echo "  Uploading ${SCRIPT_PATH} → VM:${REMOTE_SCRIPT_PATH}"
  gcloud compute scp "${SCRIPT_PATH}" "${INSTANCE}:${REMOTE_SCRIPT_PATH}" \
    --zone="${CREATED_ZONE}" --project="${PROJECT}"
else
  REL_PATH="${SCRIPT_PATH#${REPO_ROOT}/}"
  REMOTE_SCRIPT_PATH="~/src/rokin/${REL_PATH}"
fi

# Write runner script on VM (single-quoted heredoc = no expansion)
run_ssh --command="$(cat << 'LAUNCH'
cat > /tmp/run_script.sh << 'RUNSCRIPT'
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/rokin

UV_CACHE_DIR=/opt/uv-cache uv run python REMOTE_SCRIPT SCRIPT_ARGS \
  >> ~/script_output.log 2>&1
echo $? > ~/script_output.exitcode

# Upload results to GCS
gsutil cp ~/script_output.log "GCS_PREFIX_PLACEHOLDER/script_output.log"
gsutil cp ~/script_output.exitcode "GCS_PREFIX_PLACEHOLDER/script_output.exitcode"
RUNSCRIPT
chmod +x /tmp/run_script.sh
LAUNCH
)"

# Inject variables into the runner script
run_ssh --command="sed -i 's|REMOTE_SCRIPT|${REMOTE_SCRIPT_PATH}|g; s|SCRIPT_ARGS|${SCRIPT_ARGS}|g; s|GCS_PREFIX_PLACEHOLDER|${GCS_PREFIX}|g' /tmp/run_script.sh"

# Launch detached
run_ssh --command="nohup /tmp/run_script.sh > /dev/null 2>&1 & echo 'Script launched (PID '\$!')';"

# ── Step 6: Poll GCS for completion ───────────────────────────────

echo "[6/6] Polling for completion (every ${POLL_INTERVAL}s)..."
echo "  Watching: ${GCS_PREFIX}/script_output.exitcode"

while true; do
  sleep "${POLL_INTERVAL}"

  # Check VM is still alive (spot preemption)
  vm_status=$(gcloud compute instances describe "${INSTANCE}" \
    --zone="${CREATED_ZONE}" --project="${PROJECT}" \
    --format="get(status)" 2>/dev/null || echo "DELETED")

  if [[ "${vm_status}" != "RUNNING" ]]; then
    echo "  VM was preempted or stopped (status: ${vm_status})."
    echo "  Script did not complete. Re-run to retry."
    CREATED_ZONE=""  # prevent cleanup from trying to delete
    exit 1
  fi

  # Check GCS for completion marker
  if gsutil -q stat "${GCS_PREFIX}/script_output.exitcode" 2>/dev/null; then
    echo "  Script finished! Results uploaded to GCS."
    break
  fi

  echo "  Running..."
done

# ── Fetch results from GCS ────────────────────────────────────────

EXIT_CODE=$(gsutil cat "${GCS_PREFIX}/script_output.exitcode" 2>/dev/null || echo "1")

echo ""
echo "===== SCRIPT OUTPUT ====="
gsutil cat "${GCS_PREFIX}/script_output.log" 2>/dev/null
echo ""
echo "Full log: gsutil cat ${GCS_PREFIX}/script_output.log"

# VM cleanup happens via EXIT trap
exit "${EXIT_CODE}"
