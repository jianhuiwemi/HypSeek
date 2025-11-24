#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
TASK=$1                
arch=$2                 
weight_path=$3         
results_path=$4         

batch_size=256         
num_workers=8
seed=1

LOCAL_UNIMOL="$(cd "$(dirname "$0")" && pwd)/unimol"
export PYTHONPATH="${LOCAL_UNIMOL}:$PYTHONPATH"
pip uninstall -y unimol >/dev/null 2>&1 || true

DATA_ROOT=/path/test_datasets

echo "using data from ${DATA_ROOT}"
echo "writing to ${results_path}"
mkdir -p "${results_path}"

python "${LOCAL_UNIMOL}/test.py" \
    "${DATA_ROOT}" \
    --user-dir "${LOCAL_UNIMOL}" \
    --valid-subset test \
    --results-path "${results_path}" \
    --num-workers "${num_workers}" \
    --ddp-backend c10d \
    --distributed-world-size 1 \
    --batch-size "${batch_size}" \
    --task test_task \
    --loss three_hybrid_loss \
    --arch three_hybrid_model \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed "${seed}" \
    --path "${weight_path}" \
    --log-interval 100 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task "${TASK}"
