"""Bootstrap a CallIntel checkout on a fresh machine.

Examples:
    python scripts/setup.py
    python scripts/setup.py --all
    python scripts/setup.py --with-sentiment --with-diarize
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> None:
    print(f"\n> {' '.join(args)}\n")
    subprocess.run(args, cwd=ROOT_DIR, check=True)


def ensure_env_file() -> None:
    env_path = ROOT_DIR / ".env"
    example_path = ROOT_DIR / ".env.example"
    if env_path.exists() or not example_path.exists():
        return
    shutil.copyfile(example_path, env_path)
    print("Created .env from .env.example")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up CallIntel on a fresh machine.")
    parser.add_argument("--all", action="store_true", help="Install optional runtime extras and download all supported models.")
    parser.add_argument("--with-dev", action="store_true", help="Install development dependencies.")
    parser.add_argument("--with-sentiment", action="store_true", help="Download the sentiment model.")
    parser.add_argument("--with-diarize", action="store_true", help="Install diarization extras and cache the diarization model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extras: list[str] = []
    if args.with_dev or args.all:
        extras.append("dev")
    if args.with_diarize or args.all:
        extras.append("diarize")

    install_target = "."
    if extras:
        install_target = f".[{','.join(extras)}]"

    ensure_env_file()

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_command([sys.executable, "-m", "pip", "install", "-e", install_target])

    model_args = [sys.executable, "scripts/download_models.py", "--whisper"]
    if args.with_sentiment or args.all:
        model_args.append("--sentiment")
    if args.with_diarize or args.all:
        model_args.append("--diarize")
    run_command(model_args)

    print("\nSetup complete.")
    print("Required runtime pieces installed: Python dependencies + Whisper model.")
    if args.with_sentiment or args.all:
        print("Optional sentiment model installed.")
    if args.with_diarize or args.all:
        print("Optional diarization extras installed.")
    print("Place input audio in data/input_calls and run: python -m src.pipeline")


if __name__ == "__main__":
    main()
