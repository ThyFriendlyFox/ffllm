#!/usr/bin/env bash
set -euo pipefail

# Start server
python -m ffllm.federated.server --config configs/default.yaml &
SERVER_PID=$!
sleep 2

# Start clients
NUM_CLIENTS=${1:-2}
for i in $(seq 1 $NUM_CLIENTS); do
  python -m ffllm.federated.client --config configs/default.yaml &
  PIDS[$i]=$!
  sleep 1
done

wait $SERVER_PID
for pid in ${PIDS[*]}; do
  wait $pid || true
done
