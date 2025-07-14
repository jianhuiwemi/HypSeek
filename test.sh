#bash test.sh DUDE three_hybrid_model /data/protein/jianhui/hypdrugclip/save_hyper_hdr/screen_pocket_hyperbolic/savedir_screen/checkpoint_last.pt ./results/
#bash test.sh PCBA three_hybrid_model /data/protein/jianhui/hypdrugclip/save_hyper_hdr/screen_pocket_hyperbolic/savedir_screen/checkpoint_last.pt ./results/
#bash test.sh FEP three_hybrid_model /data/protein/jianhui/hypdrugclip/save_hyper_hdr_fep/screen_pocket_hyperbolic/savedir_screen/checkpoint_last.pt ./results/
#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1

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


DATA_ROOT=/data/protein/jianhui/hypdrugclip/test_datasets/test_datasets

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
    --aperture-eta 1.2 \
    --alpha-rank 1.0 \
    --alpha-corr 0.0 \
    --gamma-ce 1.0 \
    --curv-init 1.0 \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed "${seed}" \
    --path "${weight_path}" \
    --log-interval 100 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task "${TASK}"
