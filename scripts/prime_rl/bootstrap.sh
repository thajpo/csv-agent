#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
PRIME_RL_REF="${PRIME_RL_REF:-main}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$ROOT_DIR/.prime-rl/prime-rl}"
UV_BIN="${UV_BIN:-uv}"
PRIME_RL_UV_EXTRAS="${PRIME_RL_UV_EXTRAS:-flash-attn}"
PRIME_RL_UV_PIP_PACKAGES="${PRIME_RL_UV_PIP_PACKAGES:-orjson}"
CSV_AGENT_PRIME_PIP_PACKAGES="${CSV_AGENT_PRIME_PIP_PACKAGES:-datasets huggingface-hub docker pandas numpy scipy scikit-learn statsmodels}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ or set UV_BIN=/path/to/uv" >&2
  exit 1
fi

echo "Using $("$UV_BIN" --version). Prime-RL requires uv >= 0.11.1."

if [ ! -d "$PRIME_RL_DIR/.git" ]; then
  mkdir -p "$(dirname "$PRIME_RL_DIR")"
  git clone "$PRIME_RL_REPO" "$PRIME_RL_DIR"
fi

git config --global url."https://github.com/PrimeIntellect-ai/".insteadOf "git@github.com:PrimeIntellect-ai/"
git -C "$PRIME_RL_DIR" config url."https://github.com/PrimeIntellect-ai/".insteadOf "git@github.com:PrimeIntellect-ai/"
git -C "$PRIME_RL_DIR" fetch origin "$PRIME_RL_REF" --depth 1
git -C "$PRIME_RL_DIR" checkout FETCH_HEAD
git -C "$PRIME_RL_DIR" submodule update --init \
  deps/verifiers \
  deps/renderers \
  deps/research-environments \
  deps/pydantic-config

(
  cd "$PRIME_RL_DIR"
  if [ -n "$PRIME_RL_UV_EXTRAS" ]; then
    IFS=',' read -ra UV_EXTRAS <<< "$PRIME_RL_UV_EXTRAS"
    UV_SYNC_ARGS=()
    for extra in "${UV_EXTRAS[@]}"; do
      UV_SYNC_ARGS+=(--extra "$extra")
    done
    "$UV_BIN" sync "${UV_SYNC_ARGS[@]}"
  else
    "$UV_BIN" sync
  fi
  if [ -n "$PRIME_RL_UV_PIP_PACKAGES" ]; then
    # Prime-RL main currently imports orjson from the orchestrator without
    # installing it in the default sync target.
    "$UV_BIN" pip install $PRIME_RL_UV_PIP_PACKAGES
  fi
  if [ -n "$CSV_AGENT_PRIME_PIP_PACKAGES" ]; then
    # csv-agent is exposed through PYTHONPATH by run.sh rather than installed
    # into Prime-RL's Python 3.12 environment, so install only the runtime deps
    # needed by the Verifiers environment adapter.
    "$UV_BIN" pip install $CSV_AGENT_PRIME_PIP_PACKAGES
  fi
)

echo
echo "Prime-RL checkout ready: $PRIME_RL_DIR"
echo "CSV Agent will be exposed to Prime-RL by scripts/prime_rl/run.sh via PYTHONPATH."
