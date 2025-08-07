# FF-LLM: Decentralized Forward-Forward Federated LLM (Prototype)

This repo provides a minimal, runnable prototype of a decentralized federated learning network training a forward-forward (FF) modified transformer with ternary weights and FP4 activations on function-call sequences.

Key features:
- Forward-Forward learning (no backprop) with positive/negative passes
- Ternary weights (~1.58 bits) and FP4 activation quantization
- Federated learning via Flower (client/server)
- Synthetic function-call dataset
- Differential privacy noise and gradient clipping (simplified)
- Compression knobs (top-k style sparsification stub)
- Optuna-based auto-improvement orchestration

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart: Local FF training

```bash
python -m ffllm.training.run_local --config configs/default.yaml
```

## Federated (single machine simulation)
In separate terminals:

- Server:
```bash
python -m ffllm.federated.server --config configs/default.yaml
```

- Clients (run the following multiple times or in parallel):
```bash
python -m ffllm.federated.client --config configs/default.yaml
```

## Auto-Improvement (Optuna)

```bash
python -m ffllm.orchestrator.auto_improve --config configs/default.yaml --n-trials 5
```

## Notes
- Code is intentionally compact with clear entrypoints for further R&D.
- This prototype uses synthetic data and a simplified FF loss; extend as needed for real-world function-call datasets and hardware accelerators.
