"""Download pre-trained models for CallIntel.

Usage:
    python scripts/download_models.py [--whisper] [--sentiment] [--diarize] [--all]

Whisper and sentiment models are saved under ``models/``.
Pyannote diarization uses the Hugging Face cache and requires ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

MODELS = {
    "whisper": {
        "repo_id": "openai/whisper-large-v3",
        "local_dir": MODELS_DIR / "whisper-large-v3",
        "description": "Speech recognition (Whisper Large v3)",
        "size_hint": "~3.1 GB",
    },
    "sentiment": {
        "repo_id": "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment",
        "local_dir": MODELS_DIR / "sentiment",
        "description": "Arabic sentiment analysis (CAMeLBERT Mix)",
        "size_hint": "~436 MB",
    },
}


def download_model(key: str) -> None:
    from huggingface_hub import snapshot_download

    info = MODELS[key]
    print(f"\n{'=' * 60}")
    print(f"Downloading: {info['description']}")
    print(f"  Repo:  {info['repo_id']}")
    print(f"  Dest:  {info['local_dir']}")
    print(f"  Size:  {info['size_hint']}")
    print(f"{'=' * 60}\n")

    info["local_dir"].mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(info["local_dir"]),
        local_dir_use_symlinks=False,
    )
    print(f"\nOK: {key} model saved to {info['local_dir']}")


def download_diarization() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not hf_token:
        print("\nERROR: Neither HF_TOKEN nor HUGGINGFACE_TOKEN is set.")
        print("Add your Hugging Face token to .env before downloading diarization.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Downloading: Speaker Diarization (pyannote.audio)")
    print("  Repo:  pyannote/speaker-diarization-3.1")
    print("  Dest:  Hugging Face cache")
    print("  Size:  ~600 MB")
    print(f"{'=' * 60}\n")

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("\nERROR: pyannote.audio is not installed.")
        print("Install it first with: pip install -e .[diarize]")
        sys.exit(1)

    try:
        Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        print("\nOK: diarization pipeline loaded and cached successfully")
    except Exception as exc:
        print(f"\nERROR loading diarization pipeline: {exc}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models for CallIntel")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper Large v3 (~3.1 GB)")
    parser.add_argument("--sentiment", action="store_true", help="Download sentiment model (~436 MB)")
    parser.add_argument("--diarize", action="store_true", help="Download pyannote diarization (~600 MB)")
    parser.add_argument("--all", action="store_true", help="Download all supported models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    targets: list[str] = []
    if args.all:
        targets = ["whisper", "sentiment", "diarize"]
    else:
        if args.whisper:
            targets.append("whisper")
        if args.sentiment:
            targets.append("sentiment")
        if args.diarize:
            targets.append("diarize")

    if not targets:
        print("No models selected. Use --whisper, --sentiment, --diarize, or --all")
        print("\nAvailable models:")
        for key, info in MODELS.items():
            print(f"  --{key:12s} {info['description']:42s} ({info['size_hint']})")
        print(f"  --{'diarize':12s} {'Speaker diarization (HF token required)':42s} (~600 MB)")
        sys.exit(1)

    for key in targets:
        if key == "diarize":
            download_diarization()
        else:
            download_model(key)

    print(f"\nAll done. {len(targets)} model(s) processed.")


if __name__ == "__main__":
    main()
