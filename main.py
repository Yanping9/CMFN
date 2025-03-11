import argparse
import datetime
import json
import random
import time
from pathlib import Path
from option import Options
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os
from typing import NamedTuple
import clip
import wandb




class NestedTensor(NamedTuple):
    tensors: torch.Tensor
    mask: torch.Tensor



def main(args):
    utils.init_distributed_mode(args)
    #print("git:\n  {}\n".format(utils.get_sha()))

    if args.use_wandb and (not args.distributed or (args.distributed and args.rank == 0)) :
        run = wandb.init(
            project=args.dataset_file,
            name=args.model_name,
            config=vars(args),
            mode='offline'
        )#
    else:
        run = None
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print('****************')
    print(model)
    print('****************')
    ##
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.verb_aux:
        for name, p in model.named_parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if 'verb_class_embed' in name:
                p.requires_grad = True
            if 'verb_decoder' in name:
                p.requires_grad = True

    for name, p in model.named_parameters():
        if 'eval_visual_projection' in name:
            p.requires_grad = False

    if args.fix_clip:
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name:
                p.requires_grad = False

    if args.ft_clip_with_small_lr:
        if args.with_obj_clip_label and args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('visual_projection' in n or 'obj_visual_projection' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'visual_projection' in n and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_obj_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'obj_visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'obj_visual_projection' in n and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        else:
            raise

    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if args.with_clip_input:
        clip_model, _ = clip.load(args.clip_model, device)
        clip_image_encoder = clip_model.encode_image
    else:
        clip_image_encoder = None
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    if args.eval:
        test_stats = evaluate_hoi(args.dataset_file, model, clip_image_encoder, postprocessors,
                                  data_loader_val, args.subject_category_id, device, args)
        test_stats["aux_weight"] = args.aux_weight
        if args.log_path:
            Path(args.log_path).mkdir(parents=True, exist_ok=True)
            output_dir = Path(args.log_path)
            if utils.is_main_process():
                with (output_dir / "eval_log.txt").open("a") as f:
                    f.write(json.dumps(test_stats) + "\n")
        return
    if run and (not args.distributed or (args.distributed and args.rank == 0)):
        wandb.watch(model, criterion, log="all", log_freq=1000)

    print("Start training")
    start_time = time.time()
    if args.verb_aux:
        best_performance = 10000
    else:
        best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, clip_image_encoder, device, epoch, args.clip_max_norm)
        
        if run:
            train_stats['train_epoch'] = epoch
            run.log(train_stats)
        lr_scheduler.step()
        if epoch == (args.epochs - 1):
            checkpoint_path = os.path.join(output_dir, f'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        if args.verb_aux:
            performance = train_stats['verb_class_error']
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(train_stats) + "\n")
            if performance < best_performance:
                checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

                best_performance = performance
            print('verb_error:{}'.format(best_performance))
            continue
        if epoch < args.lr_drop and epoch % 5 != 0:  # eval every 5 epoch before lr_drop
            continue
        elif epoch >= args.lr_drop and epoch % 2 == 0:  # eval every 2 epoch after lr_drop
            continue

        test_stats = evaluate_hoi(args.dataset_file, model, clip_image_encoder, postprocessors,
                                  data_loader_val, args.subject_category_id, device, args)
        if run:
            test_stats['test_epoch'] = epoch
            run.log(test_stats)
            
        if args.dataset_file == 'hico':
            performance = test_stats['mAP']
            if epoch % 10 == 0 or epoch >= args.epochs-20: 
                checkpoint_path = os.path.join(output_dir, f'checkpoint_{epoch}.pth')
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']
        elif args.dataset_file == 'hoia':
            performance = test_stats['mAP']

        if performance > best_performance :
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        print("Epoch {:05d}/{:05d},best_performance {}".format(epoch + 1, args.epochs, best_performance))
        del train_stats, test_stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GSGN training and evaluation script', parents=[Options.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
