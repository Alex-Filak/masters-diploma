#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${DIPLOMA_ENV_READY:-}" ]]; then
	exec nix develop "${ROOT_DIR}" -c bash "${BASH_SOURCE[0]}" "$@"
fi

cd "${SCRIPT_DIR}"

export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

exec python facefusion.py run --execution-providers cuda --log-level info
