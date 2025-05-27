#!/bin/bash
# EXAMPLE SERVER RUN: Handles primarily CDRGE@96 runs across a 10 GPU node.

# Kill all existing screen sessions first
echo "Killing all existing screen sessions..."
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null
echo "Done cleaning up screen sessions."

# Define model sizes with updated dimensions
# tiny: ~100K params
TINY_HIDDEN=240
TINY_HEADS=12
TINY_HEAD_SIZE=20
TINY_BATCH=2048
TINY_LR=0.1

# small: ~1M params
SMALL_HIDDEN=1600
SMALL_HEADS=32
SMALL_HEAD_SIZE=50
SMALL_BATCH=2048
SMALL_LR=0.1

# medium: ~100M params
MED_HIDDEN=9600
MED_HEADS=64
MED_HEAD_SIZE=150
MED_BATCH=1024
MED_LR=0.01

# large: ~100M params
LARGE_HIDDEN=66000
LARGE_HEADS=220
LARGE_HEAD_SIZE=300
LARGE_BATCH=128 # 128*8
LARGE_LR=0.01

# # xlarge: 
# XLARGE_HIDDEN=297500
# XLARGE_HEADS=350
# XLARGE_HEAD_SIZE=850
# XLARGE_BATCH=128
# XLARGE_LR=0.001

# Base common parameters (will be overridden by model-specific params). If you want wandb:
# BASE_PARAMS="--device cuda --macro_batch_size 1 --max_iterations 10000000 --distribution rad --wandb --wandb_proj LSTM_experiments"
# if you dont want wandb
BASE_PARAMS=" --macro_batch_size 1 --max_iterations 10000000 --distribution rad"

# CDRGE@96 params (num_perturbations=96)
CDRGE_96="--num_perturbations 96"

# Make sure screen is installed
command -v screen >/dev/null 2>&1 || { echo "Screen is not installed. Please install it with: apt-get update && apt-get install -y screen"; exit 1; }

# Create screen sessions for first batch of runs
# Server 1: GPU 0 - sort + tiny + CDRGE@96
screen -dmS sort_tiny_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=0 python rge_series_experiments.py --model_type LSTM --task sort --hidden_size $TINY_HIDDEN --num_heads $TINY_HEADS --head_size $TINY_HEAD_SIZE --micro_batch_size $TINY_BATCH --learning_rate_and_epsilon $TINY_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'sort_tiny_CDRGE96_val_21to50'; exec bash"
echo "Started sort_tiny_CDRGE96 on GPU 0"

# Server 1: GPU 1 - sort + small + CDRGE@96
screen -dmS sort_small_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=1 python rge_series_experiments.py --model_type LSTM --task sort --hidden_size $SMALL_HIDDEN --num_heads $SMALL_HEADS --head_size $SMALL_HEAD_SIZE --micro_batch_size $SMALL_BATCH --learning_rate_and_epsilon $SMALL_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'sort_small_CDRGE96_val_21to50'; exec bash"
echo "Started sort_small_CDRGE96 on GPU 1"

# Server 1: GPU 2 - sort + med + CDRGE@96
screen -dmS sort_med_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=2 python rge_series_experiments.py --model_type LSTM --task sort --hidden_size $MED_HIDDEN --num_heads $MED_HEADS --head_size $MED_HEAD_SIZE --micro_batch_size $MED_BATCH --learning_rate_and_epsilon $MED_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'sort_med_CDRGE96_val_21to50'; exec bash"
echo "Started sort_med_CDRGE96 on GPU 2"

# Server 1: GPU 3 - sort + large + CDRGE@96
screen -dmS sort_large_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=3 python rge_series_experiments.py --model_type LSTM --task sort --hidden_size $LARGE_HIDDEN --num_heads $LARGE_HEADS --head_size $LARGE_HEAD_SIZE --micro_batch_size $LARGE_BATCH --learning_rate_and_epsilon $LARGE_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'sort_large_CDRGE96_val_21to50'; exec bash"
echo "Started sort_large_CDRGE96 on GPU 3"

# Server 1: GPU 4 - copy + tiny + CDRGE@96
screen -dmS copy_tiny_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=4 python rge_series_experiments.py --model_type LSTM --task copy --hidden_size $TINY_HIDDEN --num_heads $TINY_HEADS --head_size $TINY_HEAD_SIZE --micro_batch_size $TINY_BATCH --learning_rate_and_epsilon $TINY_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'copy_tiny_CDRGE96_val_21to50'; exec bash"
echo "Started copy_tiny_CDRGE96 on GPU 4"

# Server 1: GPU 5 - copy + small + CDRGE@96
screen -dmS copy_small_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=5 python rge_series_experiments.py  --model_type LSTM --task copy --hidden_size $SMALL_HIDDEN --num_heads $SMALL_HEADS --head_size $SMALL_HEAD_SIZE --micro_batch_size $SMALL_BATCH --learning_rate_and_epsilon $SMALL_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'copy_small_CDRGE96_val_21to50'; exec bash"
echo "Started copy_small_CDRGE96 on GPU 5"

# Server 1: GPU 6 - copy + med + CDRGE@96
screen -dmS copy_med_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=6 python rge_series_experiments.py --model_type LSTM --task copy --hidden_size $MED_HIDDEN --num_heads $MED_HEADS --head_size $MED_HEAD_SIZE --micro_batch_size $MED_BATCH --learning_rate_and_epsilon $MED_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'copy_med_CDRGE96_val_21to50'; exec bash"
echo "Started copy_med_CDRGE96 on GPU 6"

# Server 1: GPU 7 - copy + large + CDRGE@96
screen -dmS copy_large_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=7 python rge_series_experiments.py --model_type LSTM --task copy --hidden_size $LARGE_HIDDEN --num_heads $LARGE_HEADS --head_size $LARGE_HEAD_SIZE --micro_batch_size $LARGE_BATCH --learning_rate_and_epsilon $LARGE_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'copy_large_CDRGE96_val_21to50'; exec bash"
echo "Started copy_large_CDRGE96 on GPU 7"

# Server 1: GPU 8 - reverse + tiny + CDRGE@96
screen -dmS reverse_tiny_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=8 python rge_series_experiments.py --model_type LSTM --task reverse --hidden_size $TINY_HIDDEN --num_heads $TINY_HEADS --head_size $TINY_HEAD_SIZE --micro_batch_size $TINY_BATCH --learning_rate_and_epsilon $TINY_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'reverse_tiny_CDRGE96_val_21to50'; exec bash"
echo "Started reverse_tiny_CDRGE96 on GPU 8"

# Server 1: GPU 9 - reverse + small + CDRGE@96
screen -dmS reverse_small_CDRGE96 bash -c "CUDA_VISIBLE_DEVICES=9 python rge_series_experiments.py --task reverse --hidden_size $SMALL_HIDDEN --num_heads $SMALL_HEADS --head_size $SMALL_HEAD_SIZE --micro_batch_size $SMALL_BATCH --learning_rate_and_epsilon $SMALL_LR $BASE_PARAMS $CDRGE_96 --wandb_run_name 'reverse_small_CDRGE96_val_21to50'; exec bash"
echo "Started reverse_small_CDRGE96 on GPU 9"

echo "All first batch jobs started in detached screen sessions on Server 1!"
echo "To view a job, use: screen -r SESSION_NAME"
echo "To detach from a session without killing it, press: Ctrl+A followed by D"
echo "To list all screen sessions, use: screen -ls"
