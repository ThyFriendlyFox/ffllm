from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
import yaml

from ffllm.assistant.chat import LLMChat
from ffllm.assistant.explain import explain_training_findings

console = Console()
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_local(config_path: str) -> int:
    console.rule("Local Training")
    cmd = [sys.executable, "-m", "ffllm.training.run_local", "--config", config_path]
    return subprocess.run(cmd).returncode


def run_server(config_path: str) -> int:
    console.rule("Federated Server")
    cmd = [sys.executable, "-m", "ffllm.federated.server", "--config", config_path]
    return subprocess.run(cmd).returncode


def run_client(config_path: str) -> int:
    console.rule("Federated Client")
    cmd = [sys.executable, "-m", "ffllm.federated.client", "--config", config_path]
    return subprocess.run(cmd).returncode


def run_optuna(config_path: str, n_trials: int) -> int:
    console.rule("Auto-Improve (Optuna)")
    cmd = [sys.executable, "-m", "ffllm.orchestrator.auto_improve", "--config", config_path, "--n-trials", str(n_trials)]
    return subprocess.run(cmd).returncode


def chat_assistant(cfg_assistant_path: str | None):
    console.rule("Chat Assistant")
    default_cfg = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": None,
    }
    if cfg_assistant_path and Path(cfg_assistant_path).exists():
        with open(cfg_assistant_path, "r") as f:
            default_cfg.update(yaml.safe_load(f) or {})
    provider = Prompt.ask("Provider", default=default_cfg["provider"], choices=["openai", "anthropic", "google", "mistral", "ollama"], show_choices=True)
    model = Prompt.ask("Model", default=default_cfg["model"])
    base_url = Prompt.ask("Base URL (optional)", default=str(default_cfg["base_url"]))
    base_url = None if base_url.lower() in ("none", "") else base_url

    chat = LLMChat(provider=provider, model=model, base_url=base_url)
    console.print(Panel.fit("Type '/exit' to leave, '/clear' to clear.\nStart chatting...", title=f"{provider}:{model}"))
    messages = [{"role": "system", "content": "You are a concise helpful assistant."}]
    while True:
        user = Prompt.ask("You")
        if user.strip().lower() == "/exit":
            break
        if user.strip().lower() == "/clear":
            messages = messages[:1]
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        messages.append({"role": "user", "content": user})
        with console.status("Thinking..."):
            try:
                reply = chat.complete(messages)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                continue
        console.print(Panel.fit(reply, title="Assistant"))
        messages.append({"role": "assistant", "content": reply})


def explain_findings(config_path: str, assistant_cfg_path: str | None) -> int:
    default_cfg = {"provider": "openai", "model": "gpt-4o-mini", "base_url": None}
    if assistant_cfg_path and Path(assistant_cfg_path).exists():
        with open(assistant_cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
            default_cfg.update(data)
    provider = default_cfg["provider"]
    model = default_cfg["model"]
    base_url = default_cfg["base_url"]
    console.rule("Explain Training Findings")
    with console.status("Summarizing..."):
        try:
            summary = explain_training_findings(config_path, provider=provider, model=model, base_url=base_url)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
    console.print(Panel.fit(summary, title=f"Findings via {provider}:{model}"))
    return 0


def render_menu(config_path: str, assistant_cfg_path: str | None):
    while True:
        table = Table(title="FF-LLM Control Center", show_header=True, header_style="bold magenta")
        table.add_column("No.", justify="right", style="cyan", width=4)
        table.add_column("Action", style="white")
        table.add_row("1", "Run Local Training")
        table.add_row("2", "Start Federated Server")
        table.add_row("3", "Start Federated Client")
        table.add_row("4", "Auto-Improve (Optuna)")
        table.add_row("5", "Chat Assistant")
        table.add_row("6", "Explain Training Findings")
        table.add_row("0", "Exit")
        console.print(table)

        choice = IntPrompt.ask("Select", default=1)
        if choice == 1:
            run_local(config_path)
        elif choice == 2:
            run_server(config_path)
        elif choice == 3:
            run_client(config_path)
        elif choice == 4:
            n_trials = IntPrompt.ask("Trials", default=5)
            run_optuna(config_path, n_trials)
        elif choice == 5:
            chat_assistant(assistant_cfg_path)
        elif choice == 6:
            explain_findings(config_path, assistant_cfg_path)
        elif choice == 0:
            console.print("Goodbye!")
            break
        else:
            console.print("Invalid selection.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--assistant-config", type=str, default="configs/assistant.yaml")
    parser.add_argument("--action", type=str, default=None, choices=[None, "local", "server", "client", "optuna", "explain", "print-menu"])  # type: ignore
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    if args.action:
        # Batch mode for CI
        if args.action == "print-menu":
            table = Table(title="FF-LLM Control Center", show_header=True, header_style="bold magenta")
            table.add_column("No.", justify="right", style="cyan", width=4)
            table.add_column("Action", style="white")
            for i, name in enumerate(["Run Local Training", "Start Federated Server", "Start Federated Client", "Auto-Improve (Optuna)", "Chat Assistant", "Explain Training Findings", "Exit"], start=1):
                idx = i if i < 7 else 0
                table.add_row(str(idx), name)
            console.print(table)
            sys.exit(0)
        elif args.action == "local":
            rc = run_local(args.config)
            print("RESULT:LOCAL:", rc)
            sys.exit(rc)
        elif args.action == "server":
            rc = run_server(args.config)
            print("RESULT:SERVER:", rc)
            sys.exit(rc)
        elif args.action == "client":
            rc = run_client(args.config)
            print("RESULT:CLIENT:", rc)
            sys.exit(rc)
        elif args.action == "optuna":
            rc = run_optuna(args.config, args.n_trials)
            print("RESULT:OPTUNA:", rc)
            sys.exit(rc)
        elif args.action == "explain":
            rc = explain_findings(args.config, args.assistant_config)
            print("RESULT:EXPLAIN:", rc)
            sys.exit(rc)

    # Interactive menu
    render_menu(args.config, args.assistant_config)


if __name__ == "__main__":
    main()
