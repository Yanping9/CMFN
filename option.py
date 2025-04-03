import argparse

class Options:

    def get_args_parser():
        parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--lr_backbone', default=1e-5, type=float)
        parser.add_argument('--lr_clip', default=1e-5, type=float)
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--epochs', default=1, type=int)
        parser.add_argument('--lr_drop', default=60, type=int)
        parser.add_argument('--clip_dim', default=768, type=int)
        parser.add_argument('--clip_max_norm', default=0.1, type=float,
                            help='gradient clipping max norm')

        # Model parameters
        parser.add_argument('--frozen_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")

        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=3, type=int,
                            help="Number of stage1 decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=64, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')

        # * Segmentation
        parser.add_argument('--masks', action='store_true',
                            help="Train segmentation head if the flag is provided")

        # Visual_HOI
        parser.add_argument('--hoi', action='store_true',
                            help="Train for Visual_HOI if the flag is provided")
        parser.add_argument('--num_obj_classes', type=int, default=81,
                            help="Number of object classes")
        parser.add_argument('--num_verb_classes', type=int, default=29,
                            help="Number of verb classes")
        parser.add_argument('--pretrained', type=str, default="", help='Pretrained model path')
        parser.add_argument('--subject_category_id', default=0, type=int)
        parser.add_argument('--verb_loss_type', type=str, default='focal',
                            help='Loss type for the verb classification')

        # Loss
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument('--with_mimic', action='store_false',
                            help="Use clip feature mimic")
        # * Matcher
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=1, type=float,
                            help="giou box coefficient in the matching cost")
        parser.add_argument('--set_cost_obj_class', default=1, type=float,
                            help="Object class coefficient in the matching cost")
        parser.add_argument('--set_cost_verb_class', default=1, type=float,
                            help="Verb class coefficient in the matching cost")
        parser.add_argument('--set_cost_hoi', default=1, type=float,
                            help="Hoi class coefficient")

        # * Loss coefficients
        parser.add_argument('--mask_loss_coef', default=1, type=float)
        parser.add_argument('--dice_loss_coef', default=1, type=float)
        parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
        parser.add_argument('--giou_loss_coef', default=1, type=float)
        parser.add_argument('--obj_loss_coef', default=1, type=float)
        parser.add_argument('--verb_loss_coef', default=2, type=float)
        parser.add_argument('--hoi_loss_coef', default=2, type=float)
        parser.add_argument('--mimic_loss_coef', default=20, type=float)
        parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        # dataset parameters
        parser.add_argument('--dataset_file', default='vcoco')
        parser.add_argument('--coco_path', type=str)
        parser.add_argument('--coco_panoptic_path', type=str)
        parser.add_argument('--remove_difficult', action='store_true')
        parser.add_argument('--hoi_path', type=str, default='')
        parser.add_argument('--log_path', type=str, default='')

        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval',  action='store_true')
        parser.add_argument('--num_workers', default=2, type=int)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

        # hoi eval parameters
        parser.add_argument('--use_nms_filter', default=True, action='store_true',
                            help='Use pair nms filter, default not use')
        parser.add_argument('--thres_nms', default=0.7, type=float)
        parser.add_argument('--nms_alpha', default=1, type=float)
        parser.add_argument('--nms_beta', default=0.5, type=float)
        parser.add_argument('--logit_scale', default=9.0, type=float)
        parser.add_argument('--json_file', default='', type=str)

        # clip
        parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                            help='Use smaller learning rate to finetune clip weights')
        parser.add_argument('--with_clip_label', default=True, action='store_true', help='Use clip to classify Visual_HOI')
        parser.add_argument('--early_stop_mimic', action='store_true', help='stop mimic after step')
        parser.add_argument('--with_obj_clip_label', default=True, action='store_true',
                            help='Use clip to classify object')
        parser.add_argument('--clip_model', default='ViT-B/32',
                            help='clip pretrained model path')
        parser.add_argument('--fix_clip', action='store_true', help='')
        parser.add_argument('--clip_embed_dim', default=512, type=int)

        # zero shot type
        parser.add_argument('--zero_shot_type', default='default',
                            help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
        parser.add_argument('--del_unseen', action='store_true', help='')

        # wandb
        parser.add_argument('--use_wandb', action='store_true', help='')
        parser.add_argument('--model_name', default='cmfn', type=str)
        
        # spatial_pos
        parser.add_argument('--use_spatial_pos', action='store_false')
        parser.add_argument('--use_spatial_emd', action='store_false')
        parser.add_argument('--with_clip_input', action='store_true')

        # verb_aux
        parser.add_argument('--verb_aux', action='store_true')
        parser.add_argument('--verb_aux_eval', action='store_true')
        parser.add_argument('--aux_weight', default=0., type=float)
        return parser
