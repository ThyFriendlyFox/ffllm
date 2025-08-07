#!/usr/bin/env bash
set -euo pipefail

python -m ffllm.cli.menu --config configs/default.yaml --assistant-config configs/assistant.yaml
