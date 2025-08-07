#!/usr/bin/env bash
set -euo pipefail

TRIALS=${1:-5}
python -m ffllm.orchestrator.auto_improve --config configs/default.yaml --n-trials $TRIALS
