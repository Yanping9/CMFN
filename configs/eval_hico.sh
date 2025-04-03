#!/usr/bin/env bash

set -x
model=cmfn-
EXP_DIR=exps/training_data/hico/${model}
SAVE_DIR=exps/eval_data/hico/${model}
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained params/checkpoint_best.pth \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --log_path ${SAVE_DIR} \
        --batch_size 4 \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --model_name ${model} \
        --json_file  ${SAVE_DIR}\
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --num_workers 2 \
        --eval \
        --use_nms_filter \
        --with_clip_label \
        --with_obj_clip_label \
        --verb_aux \
        --verb_aux_eval \
        --aux_weight  0.7 
