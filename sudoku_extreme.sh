#!/bin/bash
#SBATCH --job-name=pretrain_att_dropout_sweep
#SBATCH --output=logs/pretrain_att_dropout_%A_%a.out
#SBATCH --error=logs/pretrain_att_dropout_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --array=0-0

# Dropout values to sweep
DROPOUT_RATES=(0.0 0.0 0.0 0.0 0.0)

# L-Cycle value for this job
L_cycles=(6)
L_layers=(2)
H_cycles=(3)

RATE=${DROPOUT_RATES[$SLURM_ARRAY_TASK_ID]}
L_cycle=${L_cycles[$SLURM_ARRAY_TASK_ID]}
L_layer=${L_layers[$SLURM_ARRAY_TASK_ID]}
H_cycle=${H_cycles[$SLURM_ARRAY_TASK_ID]}

# Env setup
cd /home/adagrawa/work/genai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tmr-proj

mkdir -p logs

echo "================================="
echo "Running dropout rate = ${RATE}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "================================="

run_name="pretrain_TRM_no_injection_att_sudoku_dropout_${RATE}_L_cycles_${L_cycle}_L_layers_${L_layer}_H_cycles_${H_cycle}"

torchrun --nproc-per-node 2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 pretrain.py \
    arch=trm \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.L_layers=${L_layer} \
    arch.H_cycles=${H_cycle} arch.L_cycles=${L_cycle} \
    # arch.dropout_rate=${RATE} \
    +run_name=${run_name} \
    ema=True