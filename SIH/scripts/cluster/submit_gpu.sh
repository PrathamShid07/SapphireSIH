#!/bin/bash
#SBATCH --job-name=drone_hybrid_3d
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e

mkdir -p logs

module load python/3.11.14

if [ -d "cluster_env" ]; then
    source cluster_env/bin/activate
elif [ -d "my_env" ]; then
    source my_env/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

nvidia-smi

CMD=${1:-"inspect-env"}
shift || true
ARGS="$@"

echo "Executing: python main.py $CMD $ARGS"
python main.py $CMD $ARGS
