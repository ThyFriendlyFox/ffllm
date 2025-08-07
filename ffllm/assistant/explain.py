from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import yaml

from ffllm.assistant.chat import LLMChat

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
TRAIN_LOG = LOGS_DIR / "training.jsonl"
AUTOLOG = ROOT / "AUTOLOG.md"


def _read_tail_lines(path: Path, max_lines: int = 200) -> List[str]:
    if not path.exists():
        return []
    with path.open("r") as f:
        lines = f.readlines()
    return lines[-max_lines:]


def _load_jsonl_tail(path: Path, max_lines: int = 200) -> List[Dict]:
    events: List[Dict] = []
    for line in _read_tail_lines(path, max_lines=max_lines):
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def build_context_summary(config_path: str) -> str:
    cfg_text = ""
    try:
        with open(config_path, "r") as f:
            cfg_text = f.read()
    except Exception:
        pass

    train_events = _load_jsonl_tail(TRAIN_LOG, max_lines=400)
    autolog_tail = "".join(_read_tail_lines(AUTOLOG, max_lines=100))

    summary = []
    summary.append(f"Config path: {config_path}")
    if cfg_text:
        summary.append("Config:\n" + cfg_text)
    if train_events:
        last = train_events[-1]
        epochs = [e.get("epoch") for e in train_events if "epoch" in e]
        losses = [e.get("loss") for e in train_events if "loss" in e]
        accs = [e.get("plausibility_acc") for e in train_events if "plausibility_acc" in e]
        summary.append("Training tail (last 10):\n" + "\n".join(json.dumps(e) for e in train_events[-10:]))
        if losses:
            summary.append(f"Loss trend: start={losses[0]:.4f}, end={losses[-1]:.4f}, n={len(losses)}")
        if accs:
            summary.append(f"Plausibility accuracy trend: start={accs[0]:.4f}, end={accs[-1]:.4f}, n={len(accs)}")
    else:
        summary.append("No training logs found at logs/training.jsonl")

    if autolog_tail:
        summary.append("Auto-improve tail (AUTOLOG.md):\n" + autolog_tail)

    return "\n\n".join(summary)


def explain_training_findings(
    config_path: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
) -> str:
    context = build_context_summary(config_path)
    system = (
        "You are a concise ML research assistant. Given training logs, config and optuna tail, provide: "
        "1) Key findings (loss/accuracy trends, stability, anomalies), "
        "2) Likely bottlenecks for forward-forward with ternary/FP4, "
        "3) Concrete next-step experiments (3-5), "
        "4) Risks and mitigations for federated deployment. "
        "Be specific and actionable."
    )

    # Fallback summarizer if no key available
    have_key = (
        (provider == "ollama")
        or (provider == "openai" and os.environ.get("OPENAI_API_KEY"))
        or (provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"))
        or (provider == "google" and os.environ.get("GOOGLE_API_KEY"))
        or (provider == "mistral" and os.environ.get("MISTRAL_API_KEY"))
    )

    if not have_key:
        # Simple built-in heuristic summary
        lines = context.splitlines()
        loss_start = loss_end = None
        for line in lines:
            if line.startswith("Loss trend:"):
                try:
                    parts = line.split(",")
                    loss_start = float(parts[0].split("=")[-1])
                    loss_end = float(parts[1].split("=")[-1])
                except Exception:
                    pass
        bullet = [
            "No external LLM key detected; using basic summary.",
            f"Observed loss trend: start={loss_start}, end={loss_end}",
            "Recommendations: try increasing n_layers, tune ternary_threshold (0.02-0.08), and reduce dp_sigma if convergence is slow.",
        ]
        return "\n".join([b for b in bullet if b])

    chat = LLMChat(provider=provider, model=model, base_url=base_url)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]
    return chat.complete(msgs, temperature=0.2, max_tokens=700)
