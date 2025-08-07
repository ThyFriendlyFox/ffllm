from __future__ import annotations

import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

import flwr as fl

from ffllm.data.synthetic_function_calls import FunctionCallDataset
from ffllm.models.ff_transformer import FFTransformer
from ffllm.training.forward_forward import FFConfig, train_epoch_ff


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_parameters(model: torch.nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]):
    state_dict = model.state_dict()
    for (k, _), p in zip(state_dict.items(), parameters):
        state_dict[k] = torch.tensor(p)
    model.load_state_dict(state_dict)


class FFClient(fl.client.NumPyClient):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["training"].get("device", "cpu"))
        self.model = FFTransformer(
            vocab_size=cfg["model"]["vocab_size"],
            d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"],
            n_layers=cfg["model"]["n_layers"],
            ffw_hidden=cfg["model"]["ffw_hidden"],
            max_seq_len=cfg["model"]["max_seq_len"],
            activation_bits=cfg["model"]["activation_bits"],
            ternary_threshold=cfg["model"]["ternary_threshold"],
        ).to(self.device)

        self.ds = FunctionCallDataset(
            num_sequences=cfg["data"]["num_sequences"] // 4,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            corruption_rate=cfg["data"]["corruption_rate"],
        )
        self.dl = DataLoader(self.ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["training"]["lr"]) 
        self.ff_conf = FFConfig(
            goodness_fn=cfg["forward_forward"]["goodness_fn"],
            goodness_threshold=cfg["model"]["goodness_threshold"],
            dp_sigma=cfg["privacy"]["dp_sigma"],
            clip_norm=cfg["privacy"]["clip_norm"],
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        epochs = self.cfg["federated"].get("local_epochs", 1)
        for _ in range(epochs):
            train_epoch_ff(self.model, self.dl, self.optimizer, self.ff_conf, self.device)
        return get_parameters(self.model), len(self.ds), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        # Simple proxy metric: training loss over a small batch
        batch = next(iter(self.dl))
        pos, neg = [b.to(self.device) for b in batch]
        _, pos_acts = self.model(pos)
        _, neg_acts = self.model(neg)
        loss = 0.0
        for hp, hn in zip(pos_acts, neg_acts):
            gp = (hp.pow(2).mean(dim=[1, 2])).mean()
            gn = (hn.pow(2).mean(dim=[1, 2])).mean()
            loss = loss + (gn - gp)
        return float(loss.item()), len(self.ds), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FFClient(cfg))


if __name__ == "__main__":
    main()
