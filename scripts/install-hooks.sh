#!/usr/bin/env bash
# Install a check-only pre-push gate mirroring this repo's CI lint job.
#
#   ./scripts/install-hooks.sh [--no-ty] [paths...]
#
#   --no-ty      skip the `ty check` step (for repos whose CI does not gate it)
#   paths        what to lint/format-check; defaults to the importable package
#                dir plus tests/ and scripts/ when they exist
#
# Why pre-PUSH and not pre-commit:
#   * runs once per push instead of once per commit, so WIP commits, `--amend`
#     and rebases stay fast;
#   * never rewrites files, so nothing changes under your editor mid-commit
#     (auto-fixing pre-commit setups abort and make you re-stage);
#   * mirrors CI exactly, so a green hook means a green CI lint job.
#
# It is an ACCELERATOR, not the gate. CI is the gate: .git/hooks is not
# version-controlled, is absent in a fresh clone, and `git push --no-verify`
# skips it. Never rely on it alone.
#
# Uninstall: rm .git/hooks/pre-push
set -euo pipefail

with_ty=1
if [ "${1:-}" = "--no-ty" ]; then
  with_ty=0
  shift
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [ "$#" -gt 0 ]; then
  paths="$*"
else
  # Default: the package dir (project name, dashes -> underscores) plus the
  # usual companions. Repos with a src/ layout should pass paths explicitly.
  pkg=$(sed -n 's/^name = "\(.*\)"/\1/p' pyproject.toml | head -1 | tr '-' '_')
  paths="$pkg/"
  [ -d tests ] && paths="$paths tests/"
  [ -d scripts ] && paths="$paths scripts/"
fi

hook="$repo_root/.git/hooks/pre-push"
cat > "$hook" <<HOOK
#!/usr/bin/env bash
# Auto-installed by scripts/install-hooks.sh. Mirrors this repo's CI lint job.
# Skip with: git push --no-verify
set -uo pipefail

paths="$paths"
with_ty=$with_ty
HOOK

cat >> "$hook" <<'HOOK'
fail=0

run() {
  printf '  %-18s' "$1"
  shift
  if out=$("$@" 2>&1); then
    echo "ok"
  else
    echo "FAILED"
    printf '%s\n' "$out"
    fail=1
  fi
}

echo "pre-push checks (mirrors CI):"
# shellcheck disable=SC2086
run "ruff check"   uv run --no-sync ruff check $paths
# shellcheck disable=SC2086
run "ruff format"  uv run --no-sync ruff format --check $paths
[ "$with_ty" = 1 ] && run "ty check" uv run --no-sync ty check

if [ "$fail" -ne 0 ]; then
  echo
  echo "pre-push failed. Fix with:  uv run ruff format $paths && uv run ruff check --fix $paths"
  echo "Bypass once with:           git push --no-verify"
  exit 1
fi
exit 0
HOOK

chmod +x "$hook"
echo "installed $hook  (paths: $paths, ty: $with_ty)"
