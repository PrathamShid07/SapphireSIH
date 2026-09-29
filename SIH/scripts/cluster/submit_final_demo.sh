#!/bin/bash
#SBATCH --job-name=final_demo_sparse_dense
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/final_demo/slurm_%j.out
#SBATCH --error=logs/final_demo/slurm_%j.err

set -e

mkdir -p logs/final_demo

module load python/3.11.14

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

CONFIG="config/final_demo.yaml"
OUTPUT="outputs/final_demo"

nvidia-smi
python main.py --config "$CONFIG" inspect-env

python main.py --config "$CONFIG" preprocess-video --output "$OUTPUT"
python main.py --config "$CONFIG" extract-metadata --output "$OUTPUT"
python main.py --config "$CONFIG" reconstruct-sparse --output "$OUTPUT"
python main.py --config "$CONFIG" reconstruct-dense --output "$OUTPUT"
