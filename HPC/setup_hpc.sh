#!/bin/bash
# setup_hpc.sh — run once on the HPC login node to create the virtual environment.
#
# Usage:
#   bash setup_hpc.sh

set -euo pipefail

PROJECT_DIR="$HOME/Diss-Notebook"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== HPC Environment Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "Venv dir:    $VENV_DIR"

# Create logs directory for SLURM output files
mkdir -p "$PROJECT_DIR/logs"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3.13 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists — skipping creation."
fi

source "$VENV_DIR/bin/activate"
echo "Python: $(which python) — $(python --version)"

pip install --upgrade pip ninja --quiet

# ---------------------------------------------------------------------------
# PyTorch — adjust the CUDA version to match the HPC cluster.
# Check the cluster's CUDA version with:
#   nvidia-smi            (shows driver + CUDA version on a compute node)
#   nvcc --version        (shows toolkit version if available)
#   srun -G 1 --pty bash  (open interactive shell on a GPU node to check)
#
# Common wheel URLs:
#   CUDA 11.8:  https://download.pytorch.org/whl/cu118
#   CUDA 12.1:  https://download.pytorch.org/whl/cu121
#   CUDA 12.4:  https://download.pytorch.org/whl/cu124
# ---------------------------------------------------------------------------
CUDA_WHEEL_URL="https://download.pytorch.org/whl/cu118"

echo "Installing PyTorch (CUDA wheel: $CUDA_WHEEL_URL)..."
pip install torch torchvision --index-url "$CUDA_WHEEL_URL" --quiet

echo "Installing other dependencies..."
pip install \
    timm \
    detectors \
    compressai \
    tqdm \
    pandas \
    openpyxl \
    --quiet

# openpyxl is required by pandas.DataFrame.to_excel() for .xlsx output.

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Check JPEG2000 support in Pillow:"
echo "     python -c \"from PIL import Image; Image.new('RGB',(4,4)).save('/tmp/t.jp2')\""
echo "     (If this fails, install a Pillow build with OpenJPEG support.)"
echo ""
echo "  2. Pre-download CompressAI weights (avoids download during a batch job):"
echo "      srun -G 1 --mem=8g --pty bash"
echo "      source $VENV_DIR/bin/activate"
echo "      python -c \""
echo "        from compressai.zoo import cheng2020_attn"
echo "        for q in range(1, 7): cheng2020_attn(quality=q, pretrained=True)"
echo "        print('CompressAI weights cached.')"
echo "      \""
echo ""
echo "  3. Smoke test (2 batches, CPU or GPU):"
echo "      python $PROJECT_DIR/run_pipeline.py --max-batches 2 --output /tmp/test.xlsx"
echo ""
echo "  4. Submit the full job:"
echo "      cd $PROJECT_DIR"
echo "      sbatch run_imagenet.sh"
echo "      squeue -u \$USER   # monitor progress"
