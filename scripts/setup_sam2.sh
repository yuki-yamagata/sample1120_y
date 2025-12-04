#!/usr/bin/env bash
set -euo pipefail

# Script to setup venv, install requirements, clone & install SAM2, and download checkpoints.
# Run from the sample directory: bash scripts/setup_sam2.sh

# 1. venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -V

# 2. requirements
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# 3. SAM2 clone & install
if [ ! -d sam2 ]; then
  git clone https://github.com/facebookresearch/sam2.git
fi
pip install ./sam2

# 4. checkpoints
if [ -d sam2/checkpoints ]; then
  pushd sam2/checkpoints
  if [ -x ./download_ckpts.sh ]; then
    ./download_ckpts.sh || true
  else
    echo "download_ckpts.sh not found or not executable; please download models manually."
  fi
  popd
fi

echo "Setup complete. Activate venv with: source .venv/bin/activate"
