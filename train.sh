#!/usr/bin/env bash

data_path="./data"

save_root="./save_hyper_hdr"
save_name="screen_pocket_hyperbolic"
save_dir="${save_root}/${save_name}/savedir_screen"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir_screen"
tsb_dir="${save_root}/${save_name}/tsb_dir_screen"
mkdir -p "${save_dir}"
mkdir -p "${tmp_save_dir}"
mkdir -p "${tsb_dir}"
mkdir -p "${save_root}/train_log"

n_gpu=4
MASTER_PORT=10068

finetune_mol_model="./pretrain/mol_pre_no_h_220816.pt"     
finetune_pocket_model="./pretrain/pocket_pre_220816.pt"   

batch_size=24
epoch=50
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-4


half_aperture_K=0.1    
aperture_eta=1.2       
alpha_rank=1.0         
alpha_corr=0.0          
gamma_ce=1.0            

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

UNICORE_TRAIN=$(which unicore-train)

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nproc_per_node=${n_gpu} \
  --master_port=${MASTER_PORT} \
  ${UNICORE_TRAIN} ${data_path} \
    --user-dir ./unimol \
    --train-subset train \
    --num-workers 8 \
    --ddp-backend c10d \
    --task train_task \
    --loss three_hybrid_loss \
    --arch three_hybrid_model \
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
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --update-freq ${update_freq} \
    --seed 1 \
    --tensorboard-logdir ${tsb_dir} \
    --log-interval 100 \
    --log-format simple \
    --disable-validation \
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
    --half-aperture-K ${half_aperture_K} \
    --aperture-eta ${aperture_eta} \
    --alpha-rank ${alpha_rank} \
    --alpha-corr ${alpha_corr} \
    --gamma-ce ${gamma_ce} \

