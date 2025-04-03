#!/usr/bin/env bash

set -x
model=cmfn-s
EXP_DIR=exps/training_data/hico/verb_fine/${model}

python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained exps/training_data/hico/${model}/checkpoint_best.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det/ \
        --model_name ${model} \
        --batch_size 8 \
        --num_workers 2 \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 30 \
        --lr_drop 20 \
        --with_clip_label \
        --with_obj_clip_label \
        --with_clip_input \
        --use_nms_filter \
        --use_wandb \
        --verb_aux