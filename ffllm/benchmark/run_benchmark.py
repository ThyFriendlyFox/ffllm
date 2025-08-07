from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ffllm.models.ff_transformer import FFTransformer
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = BENCH_DIR / "report.json"
REPORT_MD = BENCH_DIR / "report.md"


def run_cmd(args: list[str], env=None, timeout: int = 300) -> tuple[int, float, str, str]:
    start = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, time.time() - start, out, err
    return proc.returncode, time.time() - start, out, err


def param_count(config_path: str) -> int:
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    m = FFTransformer(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ffw_hidden=cfg["model"]["ffw_hidden"],
        max_seq_len=cfg["model"]["max_seq_len"],
        activation_bits=cfg["model"]["activation_bits"],
        ternary_threshold=cfg["model"]["ternary_threshold"],
    )
    return sum(p.numel() for p in m.parameters())


def write_report(data: dict):
    REPORT_JSON.write_text(json.dumps(data, indent=2))
    # Markdown summary
    lines = ["# Benchmark Report", ""]
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"## {k}")
            for kk, vv in v.items():
                lines.append(f"- {kk}: {vv}")
        else:
            lines.append(f"- {k}: {v}")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ci.yaml")
    args = parser.parse_args()

    results = {"config": args.config}

    # 1) Local training
    rc, dt, out, err = run_cmd([sys.executable, "-m", "ffllm.cli.menu", "--action", "local", "--config", args.config])
    results["local_training"] = {"rc": rc, "seconds": round(dt, 2)}

    # 2) Federated - server and two clients
    server = subprocess.Popen([sys.executable, "-m", "ffllm.cli.menu", "--action", "server", "--config", args.config], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)
    c1 = subprocess.Popen([sys.executable, "-m", "ffllm.cli.menu", "--action", "client", "--config", args.config], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    c2 = subprocess.Popen([sys.executable, "-m", "ffllm.cli.menu", "--action", "client", "--config", args.config], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = time.time()
    c1.wait(timeout=180)
    c2.wait(timeout=180)
    # Server exits after rounds complete; give a bit of time
    server_rc = server.wait(timeout=180)
    results["federated_round"] = {"server_rc": server_rc, "seconds": round(time.time() - start, 2)}

    # 3) Explain findings
    rc, dt, out, err = run_cmd([sys.executable, "-m", "ffllm.cli.menu", "--action", "explain", "--config", args.config])
    results["explain"] = {"rc": rc, "seconds": round(dt, 2), "summary_preview": out[:200]}

    # 4) Auto-improve
    rc, dt, out, err = run_cmd([sys.executable, "-m", "ffllm.cli.menu", "--action", "optuna", "--config", args.config, "--n-trials", "1"])
    results["auto_improve"] = {"rc": rc, "seconds": round(dt, 2)}

    # 5) Model footprint
    results["model"] = {"params": param_count(args.config)}

    write_report(results)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
