#!/usr/bin/env bash
set -euo pipefail

python -m ffllm.training.run_local --config configs/default.yaml
