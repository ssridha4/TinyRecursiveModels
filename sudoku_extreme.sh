#!/bin/bash
#SBATCH --job-name=pretrain_att_maze30x30
#SBATCH --output=logs/pretrain_att_maze30x30_%j.out
#SBATCH --error=logs/pretrain_att_maze30x30_%j.err
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=general

# Set conda environment
cd /home/adagrawa/work/genai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tmr-proj

# Create logs directory if it doesn't exist
mkdir -p logs
export WANDB_API_KEY="700b12366796842d647c1443c56edddc253bd508"

# Run the training command
run_name="pretrain_att_sudoku"
python --nproc-per-node 4 \
 --rdzv_backend=c10d \
 --rdzv_endpoint=localhost:0 \
 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} \
ema=True