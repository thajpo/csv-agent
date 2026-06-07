#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$ROOT_DIR/.prime-rl/prime-rl}"
CONFIG_PATH="${1:-$ROOT_DIR/configs/prime_rl/csv-agent-hf.toml}"
UV_BIN="${UV_BIN:-uv}"

if [ $# -gt 0 ]; then
  shift
fi

if [ ! -d "$PRIME_RL_DIR/.git" ]; then
  echo "Prime-RL checkout not found at $PRIME_RL_DIR" >&2
  echo "Run: bash scripts/prime_rl/bootstrap.sh" >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Prime-RL config not found: $CONFIG_PATH" >&2
  exit 1
fi
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ or set UV_BIN=/path/to/uv" >&2
  exit 1
fi

RUN_NAME="${CSV_AGENT_PRIME_RUN_NAME:-csv-agent-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${CSV_AGENT_PRIME_OUTPUT_DIR:-$ROOT_DIR/artifacts/prime_rl/runs/$RUN_NAME}"
ENV_FILE="${CSV_AGENT_PRIME_ENV_FILE:-$ROOT_DIR/configs/prime_rl/secrets.env}"
mkdir -p "$OUTPUT_DIR"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "Loaded environment variables from $ENV_FILE"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/packages/csv-spec/src:${PYTHONPATH:-}"

(
  cd "$PRIME_RL_DIR"
  "$UV_BIN" run --no-sync python - <<'PY'
import csv_agent
import csv_spec

print(f"csv_agent: {csv_agent.__file__}")
print(f"csv_spec: {csv_spec.__file__}")
PY
  "$UV_BIN" run --no-sync rl @ "$CONFIG_PATH" --output-dir "$OUTPUT_DIR" "$@"
)

echo
echo "Prime-RL output dir: $OUTPUT_DIR"
echo "Summarize after the run with:"
echo "  uv run python scripts/prime_rl/plot_run.py --run-dir '$OUTPUT_DIR'"
