"""Unified CLI for test/script/bench on ephemeral GPU VMs (Fire CLI).

Usage:
    python -m wzk.gcp.run test
    python -m wzk.gcp.run test --gpu_type=t4
    python -m wzk.gcp.run script /tmp/my_script.py --arg1 value1
    python -m wzk.gcp.run bench
    python -m wzk.gcp.run bench --gpu_type=t4
"""

from __future__ import annotations

import os
import subprocess
import sys

import fire

from wzk.logger import setup_logger
from wzk.time2 import get_timestamp

from ._config import DEFAULT_PROJECT, ROBOT_ZOO_REPO, ROKIN_REPO, UV_CACHE_DIR
from ._run import run_on_ephemeral_vm

logger = setup_logger(__name__)


def _get_branch() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _clone_and_setup_script(branch: str) -> str:
    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/src
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
rm -rf ~/src/rokin ~/src/robot_zoo
git clone --depth 1 --branch {branch} {ROKIN_REPO} ~/src/rokin
git clone --depth 1 {ROBOT_ZOO_REPO} ~/src/robot_zoo
cd ~/src/rokin
UV_CACHE_DIR={UV_CACHE_DIR} uv sync --dev
nvidia-smi
UV_CACHE_DIR={UV_CACHE_DIR} uv run python -c 'import jax; print("JAX backend:", jax.default_backend())'
"""


_TEST_PATHS = [
    "tests/backends/cuda/test_cuda_ffi.py",
    "tests/backends/cuda/test_cuda_codegen_targets.py",
    "tests/backends/test_parity.py",
    "tests/backends/test_jit_vmap.py",
    "tests/backends/jax/test_fk.py",
    "tests/backends/jax/test_hessian.py",
    "tests/integration/test_metadata_parity.py",
]


def _test_run_script(gcs_prefix: str) -> str:
    test_paths = " \\\n  ".join(_TEST_PATHS)
    return f"""\
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/rokin

nvidia-smi >> ~/gpu_test.log 2>&1
echo "" >> ~/gpu_test.log

UV_CACHE_DIR={UV_CACHE_DIR} uv run pytest -v --tb=long \\
  {test_paths} \\
  >> ~/gpu_test.log 2>&1
echo $? > ~/gpu_test.exitcode

gsutil cp ~/gpu_test.log "{gcs_prefix}/output.log"
gsutil cp ~/gpu_test.exitcode "{gcs_prefix}/exitcode"
"""


def _script_run_script(remote_script_path: str, script_args: str, gcs_prefix: str) -> str:
    return f"""\
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/rokin

UV_CACHE_DIR={UV_CACHE_DIR} uv run python {remote_script_path} {script_args} \\
  >> ~/script_output.log 2>&1
echo $? > ~/script_output.exitcode

gsutil cp ~/script_output.log "{gcs_prefix}/output.log"
gsutil cp ~/script_output.exitcode "{gcs_prefix}/exitcode"
"""


def _bench_run_script(branch: str) -> str:
    return f"""\
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/rokin

push_results() {{
  cd ~/src/rokin
  git add -f tests/benchmarks/results/ gpu_bench_*.txt 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "Benchmark results $(date -u +%Y-%m-%dT%H:%M:%S) $(hostname)"
    git push origin HEAD || echo "WARNING: git push failed"
  fi
}}

git config user.email "bench@rokin"
git config user.name "Benchmark Bot"

UV_CACHE_DIR={UV_CACHE_DIR} uv sync --dev
nvidia-smi

echo ""
echo "=== Internal benchmark: FK amortized throughput (CUDA) ==="
cd ~/src/rokin/tests/benchmarks
UV_CACHE_DIR={UV_CACHE_DIR} uv run python bench_rokin_amortized.py \\
  --batch-sizes 1,4,16,64,256,1024,4096,16384,32768 \\
  --min-chain 32 --modes fwd,bwd \\
  --output-dir ~/src/rokin/tests/benchmarks/results/gcp_l4 2>&1 | tee ~/src/rokin/gpu_bench_amortized.txt
cd ~/src/rokin
push_results

echo ""
echo "=== External benchmark: rokin vs cuRobo ==="
UV_CACHE_DIR={UV_CACHE_DIR} uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu130 2>/dev/null || true
UV_CACHE_DIR={UV_CACHE_DIR} uv pip install --no-build-isolation \\
  'nvidia-curobo @ git+https://github.com/NVlabs/curobo.git' 2>/dev/null || \\
  echo "cuRobo installation failed — skipping cuRobo benchmark"
UV_CACHE_DIR={UV_CACHE_DIR} uv run python tests/benchmarks/bench_curobo.py 2>&1 | tee gpu_bench_vs_curobo.txt || \\
  echo "cuRobo benchmark failed (likely not installed)"
push_results

echo ""
echo "=== All benchmarks complete. Results pushed to git. ==="
"""


def test(
    *,
    image: str,
    gpu_type: str = "l4",
    zone: str | None = None,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Run GPU tests on an ephemeral VM."""
    branch = _get_branch()
    ts = get_timestamp()
    vm_name = f"rokin-gpu-test-{ts}"
    gcs_prefix = f"gs://{os.environ['GCS_BUCKET']}/rokin/gpu-tests/{vm_name}"

    exit_code = run_on_ephemeral_vm(
        vm_name=vm_name,
        setup_script=_clone_and_setup_script(branch),
        run_script=_test_run_script(gcs_prefix),
        gpu_type=gpu_type,
        project=project,
        zone=zone,
        image=image,
        gcs_prefix=gcs_prefix,
        completion_marker="exitcode",
    )
    sys.exit(exit_code)


def script(
    script_path: str,
    *args: str,
    image: str,
    gpu_type: str = "l4",
    zone: str | None = None,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Run an arbitrary Python script on an ephemeral GPU VM."""
    branch = _get_branch()
    ts = get_timestamp()
    vm_name = f"rokin-gpu-run-{ts}"
    gcs_prefix = f"gs://{os.environ['GCS_BUCKET']}/rokin/gpu-scripts/{vm_name}"
    script_args = " ".join(args)

    # Use the script path relative to the repo on the VM
    remote_script_path = script_path

    exit_code = run_on_ephemeral_vm(
        vm_name=vm_name,
        setup_script=_clone_and_setup_script(branch),
        run_script=_script_run_script(remote_script_path, script_args, gcs_prefix),
        gpu_type=gpu_type,
        project=project,
        zone=zone,
        image=image,
        gcs_prefix=gcs_prefix,
        completion_marker="exitcode",
    )
    sys.exit(exit_code)


def bench(
    *,
    image: str,
    gpu_type: str = "l4",
    zone: str = "us-central1-a",
    project: str = DEFAULT_PROJECT,
    delete_on_exit: bool = False,
) -> None:
    """Run benchmarks on an ephemeral GPU VM."""
    branch = _get_branch()
    ts = get_timestamp()
    vm_name = f"rokin-gpu-bench-{ts}"

    exit_code = run_on_ephemeral_vm(
        vm_name=vm_name,
        setup_script=_clone_and_setup_script(branch),
        run_script=_bench_run_script(branch),
        gpu_type=gpu_type,
        project=project,
        zone=zone,
        image=image,
        gcs_prefix=None,
        delete_on_exit=delete_on_exit,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    fire.Fire({
        "test": test,
        "script": script,
        "bench": bench,
    })
