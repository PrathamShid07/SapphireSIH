#!/bin/bash
#SBATCH --job-name=drone_photogrammetry
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_stages1to5_%j.out
#SBATCH --error=logs/slurm_stages1to5_%j.err

set -e

mkdir -p logs

module load python/3.11.14

# Include colmap_env binaries & libraries if present
if [ -d "$HOME/colmap_env" ]; then
    export PATH="$HOME/colmap_env/bin:$PATH"
    export LD_LIBRARY_PATH="$HOME/colmap_env/lib:$LD_LIBRARY_PATH"
fi

if [ -d "cluster_env" ]; then
    source cluster_env/bin/activate
elif [ -d "my_env" ]; then
    source my_env/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

VIDEO_PATH=${1:-"data/input/sample_drone.mp4"}
OUTPUT_BASE=${2:-"outputs/run1"}

echo "============================================"
echo " UAV Digital Twin - Stages 2-5 (GPU)"
echo " Input : $VIDEO_PATH"
echo " Output: $OUTPUT_BASE"
echo "============================================"

if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: Video not found at $VIDEO_PATH"
    exit 1
fi

nvidia-smi
python main.py inspect-env

echo ""
echo "[Stage 2] Frame extraction & preprocessing..."
python main.py preprocess-video --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "[Stage 3] GPS/IMU metadata extraction..."
python main.py extract-metadata --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "[Stage 4] COLMAP sparse reconstruction (SfM)..."
python main.py reconstruct-sparse --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "[Stage 5] Dense MVS reconstruction..."
python main.py reconstruct-dense --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "============================================"
echo " Stages 2-5 complete."
echo " Outputs in: $OUTPUT_BASE"
echo " Next: run Stage 6+ via submit_stages6to12.sh"
echo "============================================"
