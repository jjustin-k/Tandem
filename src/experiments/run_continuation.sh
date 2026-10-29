#!/usr/bin/env bash
# Continue the existing Ada comparison, or initialize a missing arm.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
umask 077

if pgrep -u "$(id -u)" -f '[c]alibrate_cache_attention.py' >/dev/null; then
  echo "Your calibration job is still running. Let it finish before running this launcher."
  exit 1
fi

run_home="$HOME/tandem-cluster/continuation-ada"
scratch=$(cat "$run_home/scratch-path.txt")
test -d "$scratch/data" || { echo "Existing Ada cache is missing."; exit 1; }
export UV_CACHE_DIR="$scratch/uv"
export UV_PYTHON_INSTALL_DIR="$scratch/python"
export HF_HUB_CACHE="$scratch/hub"
export HF_XET_CACHE="$scratch/xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
for tokens in 1 32; do
  test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 512 || {
    echo "GPU 0 is occupied. Stopping."
    exit 1
  }
  work="$run_home/full-tokens-$tokens"
  resume=()
  if [ -d "$work/outputs" ]; then
    resume=(--resume)
  fi
  CUDA_VISIBLE_DEVICES=0 uv run --locked --script calibrate_cache_attention.py \
    --work "$work" --data-root "$scratch/data" \
    --tokens "$tokens" --epochs 30 --patience 5 --hours 2 "${resume[@]}" \
    --upload-prefix "experiments/continuation-ada-full-tokens-$tokens-20261029" \
    2>&1 | tee -a "$run_home/full-tokens-$tokens.log"
done
