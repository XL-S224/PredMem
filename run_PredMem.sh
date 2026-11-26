#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PM_DPLM_PYTHON="${PM_DPLM_PYTHON:-$PWD/.venv-dplm/bin/python}"
export PM_ESM3_PYTHON="${PM_ESM3_PYTHON:-$PWD/.venv-esm3/bin/python}"
export PM_EVAL_PYTHON="${PM_EVAL_PYTHON:-$PM_DPLM_PYTHON}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PM_RUNNER_PYTHON:-$PM_DPLM_PYTHON}" -m PredMem.runner "$@"
