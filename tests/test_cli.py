import subprocess, sys


def test_cli_print_menu():
    rc = subprocess.run([sys.executable, "-m", "ffllm.cli.menu", "--action", "print-menu"], capture_output=True, text=True).returncode
    assert rc == 0


def test_cli_local_ci():
    rc = subprocess.run([sys.executable, "-m", "ffllm.cli.menu", "--action", "local", "--config", "configs/ci.yaml"], capture_output=True, text=True).returncode
    assert rc == 0
