#!/bin/bash

##############################################################################
# Run DNC or LSTM sweeps over HPPs
##############################################################################

# 1) Kill existing screen sessions with prefix "MeZO_"
echo "[INFO] Killing existing screen sessions named '*'..."
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting runs..."


########## RUN PREFIX #############
RUN_PREFIX="overfit_experiments" 


# Define hyperparameter arrays (adjust these values as needed):
TASKS=("copy") # can use copy, reverse, sort, add, penn_treebank, etc..
ARCHITECTURES=("DNC") # can use DNC and LSTM only right now

MODEL_SCALES=(1) # Can choose anything here from 1 to 64: MODEL_SCALES=(1 2 4 8 16 32 64) 
hidden_size=128
memory_size=128
head_size=128
num_heads=1
input_dim=32
variance_reduction=1.

INPUT_SAMPLE_LENGTHS=(100)
MICRO_BATCH_SIZES=(1)
MACRO_BATCH_SIZES=(1)

LEARNING_RATES=(0.1 0.01 0.001)  # sweep LRs from 1e-5 to .1 here 
MAX_NUMS=(120)
# WARMUP_STEPS=(0) TO IMPLEMENT
WEIGHT_DECAYS=(0.)
GRAD_CLIPS=(0)


NUM_PERTURBATIONS=(8)
# NUM_PERTURBATIONS=(8 96 512)

OVERFITS=(true)

# Other configurations:
LOG_INTERVAL=100
MAX_ITERS=1000000

BPTT=false # true if you want to run baseline experiments with BPTT, or false to run ZOO.
ADAM=false # true if you want to use adam and BPTT, otherwise set to false. Not implemented for ZOO.
WANDB=false # true if you want to log results to wandb

WANDB_PROJ=""

# Function to truncate a long session name (if needed)
truncate_name() {
  echo "$1" | cut -c1-65
}

# --- Detect Available GPUs ---
echo "[INFO] Detecting available GPUs..."
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected by nvidia-smi. Exiting."
    exit 1
fi
echo "[INFO] Detected ${NUM_GPUS} GPUs with IDs: ${GPU_IDS[@]}"

run_counter=0

# --- Loop over the original hyperparameters ---
for TASK in "${TASKS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
            for INPUT_SAMPLE_LENGTH in "${INPUT_SAMPLE_LENGTHS[@]}"; do
                for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
                    for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
                        for LR in "${LEARNING_RATES[@]}"; do
                            for MAX_NUM in "${MAX_NUMS[@]}"; do
                                for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                                    for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
                                        for OVERFIT in "${OVERFITS[@]}"; do
                                              
                                              # --- Compute model scale–dependent parameters ---
                                              # this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                              # this_memory_size=$(( memory_size * MODEL_SCALE ))
                                              # this_head_size=$(( head_size * MODEL_SCALE ))
                                              # # this_num_head=$(( num_heads * MODEL_SCALE ))
                                              # this_num_head=$(( num_heads * 1 ))
                                              # this_input_size=$(( input_dim * MODEL_SCALE ))
                                              this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                              this_memory_size=$(( memory_size * MODEL_SCALE ))
                                              this_head_size=$(( head_size * MODEL_SCALE ))
                                              # this_num_head=$(( num_heads * MODEL_SCALE ))
                                              this_num_head=$(( num_heads * 1 ))
                                              this_input_size=$(( input_dim * 1 ))
                                              
                                              # --- Loop over new ablation parameters ---
                                              for numPert in "${NUM_PERTURBATIONS[@]}"; do
                                                  
                                                if [ "$BPTT" = true ]; then
                                                  EXTRA_FLAGS+=" --use_bptt"
                                                fi
                                                if [ "$ADAM" = true ]; then
                                                  EXTRA_FLAGS+=" --use_adam"
                                                fi
                                                if [ "$OVERFIT" = true ]; then
                                                  EXTRA_FLAGS+=" --use_bptt"
                                                fi
                                                
                                                if [ "$ADAPTIVE" = true ]; then
                                                  EXTRA_FLAGS+=" --adaptive"
                                                fi

                                                if [ "$WANDB" = true ]; then
                                                  EXTRA_FLAGS+=" --wandb"
                                                  EXTRA_FLAGS+=" --wandb_proj ${WANDB_PROJ}"
                                                  EXTRA_FLAGS+=" --wandb_run_name ${RUN_NAME_BASE}"
                                                fi

                                                    
                                                     RUN_NAME_BASE="${RUN_PREFIX}_${ARCH}_pert_${numPert}_scale${MODEL_SCALE}_BPTT${BPTT}_lr${LR}_overfit_${OVERFIT}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}"
    
                                                # --- Assign GPU for this run ---
                                                gpu_index=$(( run_counter % NUM_GPUS ))
                                                assigned_gpu_id=${GPU_IDS[$gpu_index]}
                                                device_string="cuda:${assigned_gpu_id}" # Device string for python script
                                                # --- END GPU Assignment ---
                                                
                                                RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                                echo "[INFO] Launching screen session: $RUN_NAME_BASE"
                                                
                                                screen -dmS "$RUN_NAME" bash -c "
                                                echo '[INFO] Starting run: $RUN_NAME';
                                                export WANDB_RUN_NAME=$RUN_NAME;
                                                python rge_series_experiments.py \
                                                      --model_type ${ARCH} \
                                                      --device ${device_string} \
                                                      --task ${TASK} \
                                                      --seq_length ${INPUT_SAMPLE_LENGTH} \
                                                      --hidden_size ${this_hidden_size} \
                                                      --memory_size ${this_memory_size} \
                                                      --head_size ${this_head_size} \
                                                      --num_heads ${this_num_head} \
                                                      --input_size ${this_input_size} \
                                                      --micro_batch_size ${MICRO_BS} \
                                                      --macro_batch_size ${MACRO_BS} \
                                                      --max_iterations ${MAX_ITERS} \
                                                      --log_interval ${LOG_INTERVAL} \
                                                      --learning_rate_and_epsilon ${LR} \
                                                      --max_num ${MAX_NUM} \
                                                      --grad_clip ${GRAD_CLIP} \
                                                      --num_perturbations ${numPert} \
                                                      --tokenizer char_level \
                                                      --distribution rad \
                                                      ${EXTRA_FLAGS} \
                                                      ;
                                                echo '[INFO] Finished run: $RUN_NAME_BASE';
                                                exec bash
                                                "
                                                run_counter=$(( run_counter + 1 ))
                                                sleep 1
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "[INFO] Done launching all screen sessions."
