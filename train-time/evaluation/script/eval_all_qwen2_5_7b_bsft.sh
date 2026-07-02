#!/bin/bash -l
#SBATCH --job-name=eval-qwen2_5_7b_bsft
#SBATCH --error=log/eval-qwen2_5_7b_bsft.%j.out
#SBATCH --output=log/eval-qwen2_5_7b_bsft.%j.out
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --qos=wangchanx
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=4-00:00:00

ulimit -u

export RAY_TMPDIR=/tmp/ray
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="/ist-project/scads/jenta/scale-pot/retool/checkpoint/multiturn-bsft-nt-qwen-2.5-7b-instruct"
CACHE_PATH="/ist-project/scads/jenta/scale-pot/retool/cache"
TARGET_PATH="${EVAL_DIR}/results/qwen2_5_7b_bsft"
PARALLEL_SIZE=1
PROMPT_TEMPLATE="${EVAL_DIR}/prompt_template.json"

# Stage: bsft — BSFT checkpoint — injects cognitive behavior

DATASETS=("AIME24" "AIME25" "MATH500")

mkdir -p "${EVAL_DIR}/log" "${TARGET_PATH}"

for DATA_NAME in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Starting evaluation for dataset: ${DATA_NAME}"
    echo "=========================================="

    apptainer exec --nv --bind /dev/shm:/dev/shm --writable-tmpfs "${EVAL_DIR}/container/custom_img_eval_retool.sif" /bin/bash -c \
        "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:\${LD_LIBRARY_PATH} \
        && cd ${EVAL_DIR} \
        && nvidia-smi \
        && python eval.py \
            --data_name ${DATA_NAME} \
            --target_path ${TARGET_PATH} \
            --model_name_or_path ${MODEL_PATH} \
            --paralle_size ${PARALLEL_SIZE} \
            --prompt_template ${PROMPT_TEMPLATE} \
            --max_tokens 16384 \
            --cache_dir ${CACHE_PATH} \
            --prompt retool_fix \
            --exe_code
        "

    echo "Completed evaluation for dataset: ${DATA_NAME}"
    echo ""
done

echo "All dataset evaluations completed!"
