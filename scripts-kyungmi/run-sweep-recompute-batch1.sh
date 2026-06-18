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

mkdir -p "$OUTPUT_DIR"

for RECOMPUTE_LEN in 1024 2048 3072 4096 5120 6144 7168 8192; do
    echo "=== Running recompute_len=${RECOMPUTE_LEN} ==="

    TRACES_DIR="${OUTPUT_DIR}/prompt_8192_bs1_nbs2_recompute_${RECOMPUTE_LEN}_sep"
    mkdir -p "$TRACES_DIR"

    echo "$SUDO_PASS" | sudo -S numactl \
            --cpunodebind="${CPU_BIND}" \
            --membind="${CPU_BIND}" \
            env CUDA_VISIBLE_DEVICES="${GPU_BIND}" \
        ~/miniconda3/envs/flexgen_env/bin/python3 \
        flexllmgen/flex_opt_kvpr.py \
        --gen-len 16 \
        --prompt-len 8192 \
        --profile \
        --model facebook/opt-6.7b \
        --percent 100 0 0 100 100 0 \
        --save-to "${TRACES_DIR}" \
        --sep-layer true \
        --gpu-batch-size 1 \
        --num-gpu-batches 2 \
        --recompute-len "${RECOMPUTE_LEN}"
        
    if [ $? -ne 0 ]; then
        echo "ERROR: Run failed for recompute_len=${RECOMPUTE_LEN}. Continuing..."
    else
        echo "Done: recompute_len=${RECOMPUTE_LEN} -> ${TRACES_DIR}"
    fi

done

echo "=== All runs complete. Results saved to: ${OUTPUT_DIR} ==="

CURRENT_USER="$(whoami)"
echo "=== Fixing ownership of ${OUTPUT_DIR} -> ${CURRENT_USER} ==="
echo "$SUDO_PASS" | sudo -S chown -R "${CURRENT_USER}:${CURRENT_USER}" "${OUTPUT_DIR}"
echo "=== Ownership updated ==="