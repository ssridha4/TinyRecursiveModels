#!/bin/bash
#SBATCH --job-name=pretrain_att_dropout_sweep
#SBATCH --output=logs/pretrain_att_dropout_%A_%a.out
#SBATCH --error=logs/pretrain_att_dropout_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --array=0-1   # <= THIS LIMITS TO 2 JOBS RUNNING IN PARALLEL

# Dropout values to sweep
DROPOUT_RATES=(0.3 0.5)
RATE=${DROPOUT_RATES[$SLURM_ARRAY_TASK_ID]}

# Env setup
cd /home/adagrawa/work/genai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tmr-proj

mkdir -p logs

echo "================================="
echo "Running dropout rate = ${RATE}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "================================="

run_name="pretrain_att_sudoku_dropout_${RATE}"

torchrun --nproc-per-node 4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 pretrain.py \
    arch=trim \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=6 \
    arch.dropout_rate=${RATE} \
    +run_name=${run_name} \
    ema=True