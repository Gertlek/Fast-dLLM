#!/bin/bash

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

eval_gsm8k() {
    local task=${1:-gsm8k}
    local length=${2:-256}
    local block_length=${3:-32}
    local num_fewshot=${4:-5}
    local factor=${5:-1.0}
    local limit=${6:-0}
    local model_path=${7:-'GSAI-ML/LLaDA-8B-Instruct'}
    local output_path=${8:-'results/'}
    
    local steps=$((length / block_length))
    
    # Build limit argument only if > 0
    local limit_arg=""
    if (( $(echo "$limit > 0" | bc -l) )); then
        limit_arg="--limit ${limit}"
    fi
    
    # dual cache+parallel factor
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} ${limit_arg} --output_path ${output_path}\
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,factor=${factor},show_speed=True
}
block_lengths=(4 8 16 32 64 128 256)
limit=0
# Call the function with default values
for bl in "${block_lengths[@]}"; do
    eval_gsm8k "gsm8k" "256" "$bl" "5" "1.0" "$limit" "GSAI-ML/LLaDA-8B-Instruct" "results/"
done