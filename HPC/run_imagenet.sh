#!/bin/bash
#SBATCH --job-name=imagenet_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -p cs
#SBATCH -G 1
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH -t 1-00:00:00
# Optional: uncomment and fill in to receive email notifications
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_EMAIL@nottingham.ac.uk

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# ---------------------------------------------------------------------------
PROJECT_DIR="$HOME/Diss-Notebook"
VENV_DIR="$PROJECT_DIR/venv"
OUTPUT_FILE="$PROJECT_DIR/results_imagenet_$SLURM_JOB_ID.xlsx"

# Set MAX_BATCHES to a small number (e.g. "50") for a dry run to check timing
# and GPU memory before committing to the full 50,000-image validation set.
# Leave empty for a full run.
MAX_BATCHES=""

# ImageNet is pre-installed on the UoN HPC cluster at this path.
IMAGENET_DIR="/data/shared/imagenet"

# Point PyTorch's model cache at the shared directory to avoid re-downloading weights.
export TORCH_HOME="/data/shared/torch"
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Start:    $(date)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-unset}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Verify PyTorch can see the GPU
python - <<'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PYEOF

# Build the command
CMD="python $PROJECT_DIR/run_pipeline.py --output $OUTPUT_FILE"

if [ -n "$MAX_BATCHES" ]; then
    CMD="$CMD --max-batches $MAX_BATCHES"
fi

if [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "ERROR: ImageNet val directory not found at $IMAGENET_DIR/val"
    exit 1
fi

CMD="$CMD --imagenet-dir $IMAGENET_DIR"

echo "Running: $CMD"
eval "$CMD"

echo "Done:  $(date)"
echo "Output: $OUTPUT_FILE"
