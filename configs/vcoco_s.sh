#!/usr/bin/env bash

set -x
model=only_cd
EXP_DIR=exps/training_data/vcoco/ablation/${model}

python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-2branch-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --model_name ${model} \
        --batch_size 4 \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --num_workers 4 \
        --epochs 100 \
        --lr_drop 60 \
        --use_nms_filter \
        --use_wandb \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --with_mimic \
        --with_clip_input \
        --mimic_loss_coef 20 
        