This project keeps large models and datasets out of the git repository.

Guidelines

- Do NOT commit the `models/` or `data/` directories; they are listed in `.gitignore`.
- Place large model files (e.g. `*.safetensors`, `*.pt`, `*.bin`) and dataset archives on a shared drive or object storage:
  - OneDrive path (example): `C:/Users/<user>/OneDrive/drive-models/`
  - External drive or NAS
  - Cloud storage (S3, GCS) — provide download script or URL
- On each machine, create the expected directories and place the files in the project root:

  - `models/` → model checkpoints
  - `data/` → dataset files and large artifacts

Optional helpers

- Provide a small `scripts/fetch_assets.sh` or `scripts/fetch_assets.ps1` to download models from a central bucket.
- Keep checksums for large files in `assets_checksums.txt` so integrity can be verified.

When cloning on another machine

1. `git clone <REPO_URL>`
2. Create a virtual environment and install dependencies: see `requirements.txt`.
3. Copy or download `models/` and `data/` into the project root as described above.

If you want, I can add a `scripts/` folder with small helper scripts to pull assets from OneDrive/S3.
