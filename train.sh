#!/usr/bin/env bash

mode=$1
if [ "$mode" == "FEP" ]; then
    valid_set="FEP"
    disable_valid="--disable-validation"
    metric_args=""     # no best-checkpoint-metric
elif [ "$mode" == "CASF" ]; then
    valid_set="CASF"
    disable_valid=""
    metric_args="--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric"
else
    echo "Usage: bash train.sh [FEP | CASF]"
    exit 1
fi


data_path="./data"

save_root="/save_root"
save_name="screen_pocket_hyperbolic"
save_dir="${save_root}/${save_name}/savedir_screen"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir_screen"
tsb_dir="${save_root}/${save_name}/tsb_dir_screen"
mkdir -p "${save_dir}" "${tmp_save_dir}" "${tsb_dir}" "${save_root}/train_log"

n_gpu=4
MASTER_PORT=10068

finetune_mol_model="./pretrain/mol_pre_no_h_220816.pt"
finetune_pocket_model="./pretrain/pocket_pre_220816.pt"

batch_size=24
batch_size_valid=32
epoch=50
warmup=0.06
update_freq=1
lr=1e-4

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
UNICORE_TRAIN=$(which unicore-train)

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nproc_per_node=${n_gpu} \
  --master_port=${MASTER_PORT} \
  ${UNICORE_TRAIN} ${data_path} \
    --user-dir ./unimol \
    --task train_task \
    --arch three_hybrid_model \
    --loss three_hybrid_loss \
    --train-subset train \
    --valid-subset valid \
    --valid-set ${valid_set} \
    --num-workers 8 \
    --ddp-backend c10d \
    --max-pocket-atoms 256 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-8 \
    --clip-norm 1.0 \
    --lr-scheduler polynomial_decay \
    --lr ${lr} \
    --warmup-ratio ${warmup} \
    --max-epoch ${epoch} \
    --batch-size ${batch_size} \
    --batch-size-valid ${batch_size_valid} \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --update-freq ${update_freq} \
    --seed 1 \
    --tensorboard-logdir ${tsb_dir} \
    --log-interval 100 \
    --log-format simple \
    --validate-interval 1 \
    --all-gather-list-size 2048000 \
    --save-dir ${save_dir} \
    --tmp-save-dir ${tmp_save_dir} \
    --keep-best-checkpoints 8 \
    --keep-last-epochs 10 \
    --find-unused-parameters \
    --finetune-pocket-model ${finetune_pocket_model} \
    --finetune-mol-model ${finetune_mol_model} \
    --max-lignum 16 \
    --learn-curv \
    --protein-similarity-thres 1.0 \
    ${disable_valid} \
    ${metric_args} \
  > ${save_root}/train_log/train_log_${save_name}.txt 2>&1
