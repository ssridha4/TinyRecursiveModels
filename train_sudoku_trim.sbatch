#!/bin/bash
#SBATCH --job-name=pretrain_trim_sudoku
#SBATCH --output=logs/pretrain_trim_sudoku_%j.out
#SBATCH --error=logs/pretrain_trim_sudoku_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --partition=general

# Set conda environment
cd /data/user_data/ssridha4/TinyRecursiveModels
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trm_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Load WANDB API key from .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not found in .env"
fi

# Run the training command
run_name="pretrain_trim_sudoku"
torchrun --nproc-per-node 2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trim \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
