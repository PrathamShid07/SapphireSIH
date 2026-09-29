#!/bin/bash
#SBATCH --job-name=drone_stages7to12
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm_stages7to12_%j.out
#SBATCH --error=logs/slurm_stages7to12_%j.err

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
echo " UAV Digital Twin - Stages 7-12"
echo " Input : $VIDEO_PATH"
echo " Output: $OUTPUT_BASE"
echo " Assumes Stages 2-5 already completed."
echo " Stage 6 (UAVid segmentation) skipped -"
echo " none of stages 7-12 depend on its output."
echo "============================================"

DENSE_MESH="$OUTPUT_BASE/dense/dense_mesh.obj"
if [ ! -f "$DENSE_MESH" ]; then
    echo "ERROR: Dense mesh not found at $DENSE_MESH"
    echo "Run submit_stages1to5.sh first."
    exit 1
fi

nvidia-smi
python main.py inspect-env

echo ""
echo "[Stage 7] Vehicle detection & tracking..."
python main.py track-vehicles --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "[Stage 8] Dynamic object masking & clean SfM..."
python main.py mask-reconstruct --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "[Stage 9] 3D object localization..."
python main.py localize-3d --video "$VIDEO_PATH" --output "$OUTPUT_BASE"

echo ""
echo "[Stage 10] Asset library management..."
python main.py manage-assets

echo ""
echo "[Stage 10] Match assets to detections..."
python main.py match-assets --output "$OUTPUT_BASE/assets"

echo ""
echo "[Stage 11] Scene fusion..."
python main.py fuse-scene \
    --objects "$OUTPUT_BASE/localization/objects_3d.json" \
    --matched-assets "$OUTPUT_BASE/assets/matched_assets.json" \
    --dense-mesh "$DENSE_MESH" \
    --output "$OUTPUT_BASE/fusion"

echo ""
echo "[Stage 12] Texturing, UV unwrap, LOD..."
python main.py texture-clean-mesh \
    --input "$OUTPUT_BASE/fusion/fused_scene.obj" \
    --output "$OUTPUT_BASE/textured" \
    --quality all

echo ""
echo "============================================"
echo " Stages 7-12 complete."
echo " Final outputs in: $OUTPUT_BASE/textured"
echo "============================================"
