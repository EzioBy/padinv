#!/bin/bash

# Help message.
if [[ $# -lt 3 ]]; then
    echo "This script launches a job of fine-tuning StyleGAN2-ADA on" \
         "FFHQ-256 like dataset with mapping network and style affine" \
         "transformation frozen."
    echo
    echo "Usage: $0 GPUS DATASET WEIGHT_PATH [OPTIONS]"
    echo
    echo "Note: All settings are already preset for fine-tuning model trained" \
         "on FFHQ-256 with 8 GPUs. Please pass addition options, which will" \
         "overwrite the original settings, if needed."
    echo
    echo "Example: $0 8 /data/cartoon256.zip /ckpt/ffhq256.pth [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2
WEIGHT_PATH=$3

./scripts/dist_train.sh ${GPUS} stylegan2_finetune \
    --job_name='stylegan2ada_finetune' \
    --weight_path=${WEIGHT_PATH} \
    --seed=0 \
    --resolution=256 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --freeze_g_synthesis_blocks='0-6' \
    --freeze_g_affine=true \
    --freeze_g_torgb_affine=true \
    --freeze_g_keywords='mapping' \
    --total_img=5_000_000 \
    --batch_size=8 \
    --val_batch_size=16 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --g_init_res=4 \
    --latent_dim=512 \
    --d_fmaps_factor=0.5 \
    --g_fmaps_factor=0.5 \
    --d_mbstd_groups=8 \
    --g_num_mappings=8 \
    --d_lr=0.0025 \
    --g_lr=0.0025 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.9 \
    --r1_interval=16 \
    --r1_gamma=1.0 \
    --pl_interval=4 \
    --pl_weight=2.0 \
    --pl_decay=0.01 \
    --pl_batch_shrink=2 \
    --g_ema_img=20_000 \
    --g_ema_rampup=0.0 \
    --eval_at_start=true \
    --eval_interval=3200 \
    --ckpt_interval=3200 \
    --log_interval=64 \
    --enable_amp=false \
    --use_ada=true \
    --num_fp16_res=4 \
    ${@:4}
