#!/bin/bash

# Usage: ./run_flexllm_sweep.sh <output_folder> <sudo_password> <cpu_bind> <gpu_bind>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_folder> <sudo_password> <cpu_bind> <gpu_bind>"
    exit 1
fi

OUTPUT_DIR="$1"
SUDO_PASS="$2"
CPU_BIND="$3"
GPU_BIND="$4"

GPU_BATCH_SIZES=(1  2  4  8  16)
NUM_GPU_BATCHES=(16 8  4  2  1)

mkdir -p "$OUTPUT_DIR"

for PROMPT_LEN in 4096 ; do
    for i in "${!GPU_BATCH_SIZES[@]}"; do
        BS="${GPU_BATCH_SIZES[$i]}"
        NB="${NUM_GPU_BATCHES[$i]}"

        echo "=== Running prompt_len=${PROMPT_LEN}, gpu_batch_size=${BS}, num_gpu_batches=${NB} ==="

        TRACES_DIR="${OUTPUT_DIR}/prompt_${PROMPT_LEN}_bs${BS}_sep"
        mkdir -p "$TRACES_DIR"

        echo "$SUDO_PASS" | sudo -S numactl \
            --cpunodebind="${CPU_BIND}" \
            --membind="${CPU_BIND}" \
            env CUDA_VISIBLE_DEVICES="${GPU_BIND}" \
            ~/miniconda3/envs/flexgen_env/bin/python3 \
            flexllmgen/flex_opt_kvpr.py \
            --gen-len 16 \
            --prompt-len "${PROMPT_LEN}" \
            --gpu-batch-size "${BS}" \
            --num-gpu-batches "${NB}" \
            --profile \
            --model facebook/opt-6.7b \
            --percent 100 0 0 100 100 0 \
            --save-to "${TRACES_DIR}" \
            --sep-layer true

        if [ $? -ne 0 ]; then
            echo "ERROR: Run failed for prompt_len=${PROMPT_LEN}, bs=${BS}. Continuing..."
        else
            echo "Done: prompt_len=${PROMPT_LEN}, bs=${BS} -> ${TRACES_DIR}"
        fi

    done
done

echo "=== All runs complete. Results saved to: ${OUTPUT_DIR} ==="

CURRENT_USER="$(whoami)"
echo "=== Fixing ownership of ${OUTPUT_DIR} -> ${CURRENT_USER} ==="
echo "$SUDO_PASS" | sudo -S chown -R "${CURRENT_USER}:${CURRENT_USER}" "${OUTPUT_DIR}"
echo "=== Ownership updated ==="