from __future__ import annotations

import argparse
import yaml
import flwr as fl


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    rounds = cfg["federated"]["rounds"]
    clients_per_round = cfg["federated"]["clients_per_round"]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=clients_per_round,
        min_available_clients=clients_per_round,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=rounds),
    )


if __name__ == "__main__":
    main()
