#!/bin/bash
#SBATCH --job-name=drone_full_pipeline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_pipeline_%j.out
#SBATCH --error=logs/slurm_pipeline_%j.err

set -e

mkdir -p logs

module load python/3.11.14

if [ -d "cluster_env" ]; then
    source cluster_env/bin/activate
elif [ -d "my_env" ]; then
    source my_env/bin/activate
fi

VIDEO_PATH=${1:-"data/sample_drone.mp4"}
OUTPUT_BASE=${2:-"outputs/cluster_run"}

echo "Running complete pipeline on SPIT GPU Cluster"
echo "Input video: $VIDEO_PATH"
echo "Output base: $OUTPUT_BASE"

nvidia-smi

python main.py inspect-env

python main.py preprocess-video --video "$VIDEO_PATH" --output "$OUTPUT_BASE"
python main.py extract-metadata --video "$VIDEO_PATH" --output "$OUTPUT_BASE/metadata"
python main.py reconstruct-sparse --video "$VIDEO_PATH" --output "$OUTPUT_BASE"
python main.py reconstruct-dense --video "$VIDEO_PATH" --output "$OUTPUT_BASE/dense"
python main.py track-vehicles --video "$VIDEO_PATH" --output "$OUTPUT_BASE/tracking"
python main.py mask-reconstruct --video "$VIDEO_PATH" --output "$OUTPUT_BASE/masking"
python main.py localize-3d --video "$VIDEO_PATH" --output "$OUTPUT_BASE/localization"
python main.py match-assets --output "$OUTPUT_BASE/assets"
python main.py fuse-scene --objects "$OUTPUT_BASE/localization/objects_3d.json" --matched-assets "$OUTPUT_BASE/assets/matched_assets.json" --dense-mesh "$OUTPUT_BASE/dense/dense_mesh.obj" --output "$OUTPUT_BASE/fusion"
python main.py texture-clean-mesh --input "$OUTPUT_BASE/fusion/fused_scene.obj" --output "$OUTPUT_BASE/textured"

echo "Pipeline complete. Outputs stored in $OUTPUT_BASE"
