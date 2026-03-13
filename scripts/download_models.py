"""Download pre-trained models for CallIntel.

Usage:
    python scripts/download_models.py [--sentiment] [--diarize] [--all]

Models are saved to the ``models/`` directory, except for pyannote which uses the HuggingFace cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

MODELS = {
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

    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(info["local_dir"]),
        local_dir_use_symlinks=False,
    )
    print(f"\n✓ {key} model saved to {info['local_dir']}")


def download_diarization() -> None:
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    # Check both potential token names as added to config.py
    hf_token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not hf_token:
        print("\nERROR: Neither HF_TOKEN nor HUGGINGFACE_TOKEN found in environment or .env file.")
        print("Please add your HuggingFace token to use pyannote/speaker-diarization-3.1")
        sys.exit(1)
        
    print(f"\n{'=' * 60}")
    print("Downloading: Speaker Diarization (pyannote.audio)")
    print("  Repo:  pyannote/speaker-diarization-3.1")
    print("  Dest:  HuggingFace Hub Cache")
    print("  Size:  ~600 MB")
    print(f"{'=' * 60}\n")
    
    # Importing Pipeline triggers the download of the required models
    try:
        from pyannote.audio import Pipeline
        # Aligned with user's edit to diarize.py using token= argument
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        print("\n✓ Diarization pipeline loaded and cached successfully!")
    except Exception as e:
        print(f"\nERROR loading diarization pipeline: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for CallIntel")
    parser.add_argument("--sentiment", action="store_true", help="Download sentiment model (~436 MB)")
    parser.add_argument("--diarize", action="store_true", help="Download speaker diarization model (~600 MB)")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    args = parser.parse_args()

    targets: list[str] = []
    if args.all:
        targets = list(MODELS.keys())
        targets.append("diarize")
    else:
        if args.sentiment:
            targets.append("sentiment")
        if args.diarize:
            targets.append("diarize")

    if not targets:
        print("No models selected. Use --sentiment, --diarize, or --all")
        print("\nAvailable models:")
        for key, info in MODELS.items():
            print(f"  --{key:20s} {info['description']:40s} ({info['size_hint']})")
        print(f"  --{'diarize':20s} {'Speaker Diarization (Requires HF_TOKEN)':40s} (~600 MB)")
        sys.exit(1)

    for key in targets:
        if key == "diarize":
            download_diarization()
        else:
            download_model(key)

    print(f"\nAll done! {len(targets)} model(s) downloaded.")


if __name__ == "__main__":
    main()
