from __future__ import annotations

import argparse
import yaml
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import optim

from ffllm.data.synthetic_function_calls import FunctionCallDataset
from ffllm.models.ff_transformer import FFTransformer
from ffllm.training.forward_forward import FFConfig, train_epoch_ff
from ffllm.evaluation.metrics import plausibility_accuracy

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = LOGS_DIR / "training.jsonl"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def append_event(event: dict):
    with TRAIN_LOG.open("a") as f:
        f.write(json.dumps(event) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg["training"].get("device", "cpu"))

    ds = FunctionCallDataset(
        num_sequences=cfg["data"]["num_sequences"],
        seq_len=cfg["data"]["seq_len"],
        vocab_size=cfg["data"]["vocab_size"],
        corruption_rate=cfg["data"]["corruption_rate"],
    )
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = FFTransformer(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ffw_hidden=cfg["model"]["ffw_hidden"],
        max_seq_len=cfg["model"]["max_seq_len"],
        activation_bits=cfg["model"]["activation_bits"],
        ternary_threshold=cfg["model"]["ternary_threshold"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"]) 

    ff_conf = FFConfig(
        goodness_fn=cfg["forward_forward"]["goodness_fn"],
        goodness_threshold=cfg["model"]["goodness_threshold"],
        dp_sigma=cfg["privacy"]["dp_sigma"],
        clip_norm=cfg["privacy"]["clip_norm"],
    )

    for epoch in range(cfg["training"]["epochs"]):
        metrics = train_epoch_ff(model, dl, optimizer, ff_conf, device)
        eval_metrics = plausibility_accuracy(model, dl, device)
        event = {"epoch": epoch, **metrics, **eval_metrics}
        print(event)
        append_event(event)


if __name__ == "__main__":
    main()
