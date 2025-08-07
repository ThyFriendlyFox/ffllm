from __future__ import annotations

import argparse
import yaml
import optuna
import torch
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime

from ffllm.data.synthetic_function_calls import FunctionCallDataset
from ffllm.models.ff_transformer import FFTransformer
from ffllm.training.forward_forward import FFConfig, train_epoch_ff


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(path: str, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def append_autolog(msg: str):
    with open("AUTOLOG.md", "a") as f:
        f.write(msg + "\n")


def objective(trial: optuna.Trial, base_cfg: dict) -> float:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy

    cfg["model"]["d_model"] = trial.suggest_categorical("d_model", [64, 96, 128, 192])
    cfg["model"]["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 6])
    cfg["model"]["n_layers"] = trial.suggest_int("n_layers", 1, 4)
    cfg["model"]["ffw_hidden"] = trial.suggest_categorical("ffw_hidden", [128, 256, 384])
    cfg["model"]["ternary_threshold"] = trial.suggest_float("ternary_threshold", 0.01, 0.1, log=True)
    cfg["training"]["lr"] = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    cfg["privacy"]["dp_sigma"] = trial.suggest_float("dp_sigma", 0.0, 0.02)

    device = torch.device(cfg["training"].get("device", "cpu"))

    ds = FunctionCallDataset(
        num_sequences=max(1000, cfg["data"]["num_sequences"] // 5),
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

    # Short train for speed
    for _ in range(1):
        train_epoch_ff(model, dl, optimizer, ff_conf, device)

    # Validation proxy: final batch plausibility gap
    pos, neg = next(iter(dl))
    pos, neg = pos.to(device), neg.to(device)
    with torch.no_grad():
        _, pa = model(pos)
        _, na = model(neg)
        gp = sum([(h.pow(2).mean(dim=[1, 2])) for h in pa]).mean()
        gn = sum([(h.pow(2).mean(dim=[1, 2])) for h in na]).mean()
        score = (gp - gn).item()
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    study = optuna.create_study(direction="maximize")
    best = study.optimize(lambda t: objective(t, base_cfg), n_trials=args.n_trials)

    best_params = study.best_params
    timestamp = datetime.utcnow().isoformat()
    append_autolog(f"[{timestamp}] Best params: {best_params} | value={study.best_value:.4f}")

    # Persist an updated config under configs/best.yaml
    best_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    for k, v in best_params.items():
        if k in best_cfg.get("model", {}):
            best_cfg["model"][k] = v
        elif k in best_cfg.get("training", {}):
            best_cfg["training"][k] = v
        elif k in best_cfg.get("privacy", {}):
            best_cfg["privacy"][k] = v
    dump_yaml("configs/best.yaml", best_cfg)
    append_autolog(f"[{timestamp}] Saved configs/best.yaml")


if __name__ == "__main__":
    main()
