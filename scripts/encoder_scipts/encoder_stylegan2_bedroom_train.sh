#!/bin/bash

# Help message.
if [[ $# -lt 4 ]]; then
    echo "This scripts launches a job of training a PadInv encoder for" \
         "StyleGAN2 on LSUN-Bedroom, with generated image resolution at 256x256."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 \$GPUS \$TRAIN_DATASET \$VAL_DATASET \$GAN_CKPT [\$JOB_NAME]"
    echo
    echo "Example: $0 8 ~/data/bedroom_train_lmdb ~/data/bedroom_val_lmdb ~/data/stylegan2_bedroom_256_60000.pth"
    echo
    exit 0
fi

GPUS=$1
TRAIN_DATASET=$2
VAL_DATASET=$3
GAN_CKPT=$4

bash ./scripts/dist_train.sh ${GPUS} padinv_encoder \
    --job_name='padinv_bedroom' \
    --seed=0 \
    --encode_resolution=256 \
    --gen_resolution=256 \
    --image_channels=3 \
    --train_dataset=${TRAIN_DATASET} \
    --val_dataset=${VAL_DATASET} \
    --pretrained_gan_path=${GAN_CKPT} \
    --val_max_samples=-1 \
    --total_img=6_400_000 \
    --batch_size=8 \
    --val_batch_size=8 \
    --train_data_mirror=false \
    --data_loader_type='iter' \
    --data_repeat=60 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --start_from_latent_avg=true \
    --use_discriminator=true \
    --use_padding_space=true \
    --encode_const_input=true \
    --pad_layer_num=4 \
    --use_pad_mod_head=true \
    --e_lr=1e-4 \
    --mse_weight=1.0 \
    --lpips_weight=0.8 \
    --idsim_weight=0.0 \
    --wnorm_weight=0.003 \
    --adv_weight=0.05 \
    --g_init_res=4 \
    --z_dim=512 \
    --g_fmaps_factor=1.0 \
    --eval_at_start=true \
    --eval_interval=2000 \
    --ckpt_interval=2000 \
    --log_interval=100 \
    --enable_amp=false \
    --num_fp16_res=0 \
    --calc_idsim=false \
    ${@:5}
