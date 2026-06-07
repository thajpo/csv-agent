#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
PRIME_RL_REF="${PRIME_RL_REF:-main}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$ROOT_DIR/.prime-rl/prime-rl}"
UV_BIN="${UV_BIN:-uv}"

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
  "$UV_BIN" sync
)

echo
echo "Prime-RL checkout ready: $PRIME_RL_DIR"
echo "CSV Agent will be exposed to Prime-RL by scripts/prime_rl/run.sh via PYTHONPATH."
