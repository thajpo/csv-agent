#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX_BACKEND="${CSV_AGENT_SANDBOX_BACKEND:-docker}"

failures=0

section() {
  printf "\n== %s ==\n" "$1"
}

check() {
  local name="$1"
  shift
  printf "[check] %s\n" "$name"
  if "$@"; then
    printf "[ok] %s\n" "$name"
  else
    local rc=$?
    printf "[fail] %s (exit %s)\n" "$name" "$rc"
    failures=$((failures + 1))
  fi
}

section "System"
uname -a || true
command -v python >/dev/null 2>&1 && python --version || true
command -v uv >/dev/null 2>&1 && uv --version || true
command -v git >/dev/null 2>&1 && git --version || true

section "NVIDIA"
check "nvidia-smi is available" command -v nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || failures=$((failures + 1))
fi

if [ "$SANDBOX_BACKEND" = "docker" ]; then
  section "Docker"
  check "docker CLI is available" command -v docker
  if command -v docker >/dev/null 2>&1; then
    check "docker daemon is reachable" docker info
    check "docker can run a basic container" docker run --rm python:3.11-slim python -c "print('docker-ok')"

    section "CSV sandbox image"
    (
      cd "$ROOT_DIR" || exit 1
      check "csv-analysis-env image builds" docker build -t csv-analysis-env -f src/envs/Dockerfile .
      check "csv-analysis-env runs python" docker run --rm csv-analysis-env python -c "import pandas, scipy, sklearn, statsmodels; print('csv-image-ok')"
    )
  fi
else
  section "Sandbox"
  echo "[ok] using Prime-hosted sandboxes; local Docker checks skipped"
fi

section "Secrets"
for name in HF_TOKEN WANDB_API_KEY PRIME_API_KEY; do
  if [ -n "${!name:-}" ]; then
    echo "[ok] $name is set"
  else
    echo "[warn] $name is not set"
  fi
done

section "Result"
if [ "$failures" -eq 0 ]; then
  echo "Prime box preflight passed."
  exit 0
fi

echo "Prime box preflight failed with $failures failure(s)."
exit 1
