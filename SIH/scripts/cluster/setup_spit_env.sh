#!/bin/bash
set -e

module load python/3.11.14

if ! command -v uv &> /dev/null; then
    pip install --user uv
fi

uv venv cluster_env
source cluster_env/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
