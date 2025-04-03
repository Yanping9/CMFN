# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

vcoco_hoi_text_label = [{(0, 41): 'holding a cup'},
                        {(16, 80): 'cutting with something'},
                        {(17, 53): 'cutting a pizza'},
                        {(0, 53): 'holding a pizza'}, {(2, 80): 'sitting'},
                        {(8, 53): 'eating a pizza'},
                        {(9, 80): 'eating with something'},
                        {(23, 80): 'smiling'}, {(21, 37): 'surfing a surfboard'},
                        {(0, 73): 'holding a book'},
                        {(2, 13): 'sitting a bench'},
                        {(5, 73): 'looking at a book'},
                        {(27, 73): 'reading a book'}, {(1, 80): 'standing'},
                        {(22, 36): 'skateboarding a skateboard'},
                        {(20, 30): 'skiing a skis'}, {(0, 80): 'holding'},
                        {(8, 80): 'eating'}, {(2, 56): 'sitting a chair'},
                        {(5, 63): 'looking at a laptop'},
                        {(19, 63): 'working on computer a laptop'},
                        {(0, 40): 'holding a wine glass'},
                        {(24, 40): 'drinking a wine glass'},
                        {(5, 31): 'looking at a snowboard'},
                        {(28, 31): 'snowboarding a snowboard'},
                        {(0, 76): 'holding a scissors'},
                        {(5, 80): 'looking at something'},
                        {(5, 76): 'looking at a scissors'},
                        {(16, 76): 'cutting with a scissors'},
                        {(17, 80): 'cutting'},
                        {(5, 37): 'looking at a surfboard'},
                        {(2, 17): 'sitting a horse'},
                        {(3, 17): 'riding a horse'}, {(4, 80): 'walking'},
                        {(5, 29): 'looking at a frisbee'}, {(10, 80): 'jumping'},
                        {(14, 29): 'throwing a frisbee'}, {(18, 80): 'running'},
                        {(5, 53): 'looking at a pizza'},
                        {(0, 48): 'holding a sandwich'},
                        {(8, 48): 'eating a sandwich'},
                        {(0, 67): 'holding a cell phone'},
                        {(19, 80): 'working on computer'},
                        {(0, 24): 'holding a backpack'},
                        {(13, 24): 'carrying a backpack'}, {(11, 80): 'laying'},
                        {(11, 57): 'laying a couch'},
                        {(0, 17): 'holding a horse'}, {(0, 15): 'holding a cat'},
                        {(11, 59): 'laying a bed'},
                        {(15, 29): 'catching a frisbee'}, {(3, 80): 'riding'},
                        {(12, 67): 'talking on phone a cell phone'},
                        {(0, 31): 'holding a snowboard'},
                        {(10, 31): 'jumping a snowboard'},
                        {(5, 36): 'looking at a skateboard'},
                        {(10, 36): 'jumping a skateboard'},
                        {(0, 79): 'holding a toothbrush'}, {(27, 80): 'reading'},
                        {(0, 39): 'holding a bottle'},
                        {(24, 39): 'drinking a bottle'},
                        {(2, 59): 'sitting a bed'},
                        {(5, 48): 'looking at a sandwich'},
                        {(0, 30): 'holding a skis'},
                        {(0, 38): 'holding a tennis racket'},
                        {(5, 32): 'looking at a sports ball'},
                        {(6, 38): 'hitting with a tennis racket'},
                        {(7, 32): 'hitting a sports ball'},
                        {(5, 0): 'looking at a person'},
                        {(5, 17): 'looking at a horse'},
                        {(0, 47): 'holding an apple'},
                        {(5, 18): 'looking at a sheep'},
                        {(8, 47): 'eating an apple'},
                        {(25, 32): 'kicking a sports ball'},
                        {(0, 44): 'holding a spoon'},
                        {(5, 55): 'looking at a cake'},
                        {(8, 55): 'eating a cake'},
                        {(9, 44): 'eating with a spoon'},
                        {(0, 63): 'holding a laptop'},
                        {(6, 80): 'hitting with something'},
                        {(2, 3): 'sitting a motorcycle'},
                        {(3, 3): 'riding a motorcycle'},
                        {(0, 43): 'holding a knife'},
                        {(5, 43): 'looking at a knife'},
                        {(16, 43): 'cutting with a knife'},
                        {(17, 55): 'cutting a cake'}, {(7, 80): 'hitting'},
                        {(0, 34): 'holding a baseball bat'},
                        {(6, 34): 'hitting with a baseball bat'},
                        {(15, 80): 'catching'}, {(2, 57): 'sitting a couch'},
                        {(0, 77): 'holding a teddy bear'},
                        {(13, 49): 'carrying an orange'},
                        {(0, 42): 'holding a fork'},
                        {(9, 42): 'eating with a fork'},
                        {(5, 62): 'looking at a tv'},
                        {(0, 28): 'holding a suitcase'},
                        {(13, 28): 'carrying a suitcase'},
                        {(2, 20): 'sitting an elephant'},
                        {(3, 20): 'riding an elephant'},
                        {(5, 15): 'looking at a cat'},
                        {(0, 56): 'holding a chair'},
                        {(5, 60): 'looking at a dining table'},
                        {(24, 41): 'drinking a cup'}, {(14, 80): 'throwing'},
                        {(13, 26): 'carrying a handbag'},
                        {(5, 16): 'looking at a dog'},
                        {(0, 46): 'holding a banana'},
                        {(13, 46): 'carrying a banana'},
                        {(5, 28): 'looking at a suitcase'},
                        {(9, 43): 'eating with a knife'},
                        {(0, 37): 'holding a surfboard'},
                        {(13, 37): 'carrying a surfboard'},
                        {(8, 54): 'eating a donut'},
                        {(0, 0): 'holding a person'},
                        {(0, 35): 'holding a baseball glove'},
                        {(0, 65): 'holding a remote'},
                        {(0, 54): 'holding a donut'},
                        {(0, 26): 'holding a handbag'}, {(13, 80): 'carrying'},
                        {(13, 0): 'carrying a person'},
                        {(0, 32): 'holding a sports ball'},
                        {(14, 32): 'throwing a sports ball'},
                        {(5, 54): 'looking at a donut'},
                        {(0, 1): 'holding a bicycle'},
                        {(2, 1): 'sitting a bicycle'},
                        {(3, 1): 'riding a bicycle'},
                        {(5, 1): 'looking at a bicycle'}, {(25, 80): 'kicking'},
                        {(5, 67): 'looking at a cell phone'},
                        {(5, 6): 'looking at a train'},
                        {(0, 29): 'holding a frisbee'},
                        {(0, 36): 'holding a skateboard'},
                        {(3, 7): 'riding a truck'},
                        {(26, 63): 'pointing a laptop'},
                        {(0, 3): 'holding a motorcycle'},
                        {(13, 30): 'carrying a skis'},
                        {(0, 25): 'holding a umbrella'},
                        {(5, 45): 'looking at a bowl'},
                        {(17, 51): 'cutting a carrot'},
                        {(0, 52): 'holding a hot dog'},
                        {(8, 52): 'eating a hot dog'},
                        {(0, 33): 'holding a kite'},
                        {(5, 13): 'looking at a bench'},
                        {(12, 80): 'talking on phone'},
                        {(22, 80): 'skateboarding'},
                        {(5, 35): 'looking at a baseball glove'},
                        {(15, 32): 'catching a sports ball'},
                        {(26, 80): 'pointing'},
                        {(13, 25): 'carrying a umbrella'},
                        {(5, 40): 'looking at a wine glass'},
                        {(10, 37): 'jumping a surfboard'},
                        {(5, 33): 'looking at a kite'},
                        {(13, 33): 'carrying a kite'},
                        {(3, 6): 'riding a train'},
                        {(5, 44): 'looking at a spoon'},
                        {(0, 20): 'holding an elephant'}, {(21, 80): 'surfing'},
                        {(5, 20): 'looking at an elephant'},
                        {(3, 8): 'riding a boat'},
                        {(5, 23): 'looking at a giraffe'},
                        {(13, 67): 'carrying a cell phone'},
                        {(11, 56): 'laying a chair'},
                        {(5, 19): 'looking at a cow'},
                        {(5, 42): 'looking at a fork'},
                        {(0, 55): 'holding a cake'},
                        {(13, 32): 'carrying a sports ball'},
                        {(5, 30): 'looking at a skis'},
                        {(13, 36): 'carrying a skateboard'},
                        {(26, 67): 'pointing a cell phone'},
                        {(5, 52): 'looking at a hot dog'},
                        {(8, 46): 'eating a banana'}, {(20, 80): 'skiing'},
                        {(28, 80): 'snowboarding'}, {(0, 14): 'holding a bird'},
                        {(11, 60): 'laying a dining table'},
                        {(0, 16): 'holding a dog'},
                        {(0, 72): 'holding a refrigerator'},
                        {(5, 72): 'looking at a refrigerator'},
                        {(5, 7): 'looking at a truck'},
                        {(5, 41): 'looking at a cup'},
                        {(2, 61): 'sitting a toilet'}, {(24, 80): 'drinking'},
                        {(0, 27): 'holding a tie'},
                        {(5, 27): 'looking at a tie'},
                        {(17, 27): 'cutting a tie'},
                        {(5, 10): 'looking at a fire hydrant'},
                        {(26, 10): 'pointing a fire hydrant'},
                        {(11, 13): 'laying a bench'},
                        {(17, 18): 'cutting a sheep'},
                        {(0, 64): 'holding a mouse'},
                        {(5, 64): 'looking at a mouse'},
                        {(5, 66): 'looking at a keyboard'},
                        {(16, 42): 'cutting with a fork'},
                        {(17, 0): 'cutting a person'},
                        {(5, 5): 'looking at a bus'}, {(3, 2): 'riding a car'},
                        {(10, 30): 'jumping a skis'},
                        {(5, 4): 'looking at an airplane'},
                        {(5, 46): 'looking at a banana'},
                        {(2, 28): 'sitting a suitcase'},
                        {(13, 29): 'carrying a frisbee'},
                        {(5, 26): 'looking at a handbag'},
                        {(8, 50): 'eating a broccoli'},
                        {(17, 46): 'cutting a banana'},
                        {(0, 18): 'holding a sheep'},
                        {(17, 48): 'cutting a sandwich'},
                        {(26, 0): 'pointing a person'},
                        {(5, 3): 'looking at a motorcycle'},
                        {(5, 24): 'looking at a backpack'},
                        {(0, 45): 'holding a bowl'},
                        {(26, 27): 'pointing a tie'},
                        {(0, 49): 'holding an orange'},
                        {(8, 49): 'eating an orange'},
                        {(5, 34): 'looking at a baseball bat'},
                        {(13, 31): 'carrying a snowboard'},
                        {(17, 54): 'cutting a donut'},
                        {(5, 38): 'looking at a tennis racket'},
                        {(8, 51): 'eating a carrot'},
                        {(17, 47): 'cutting an apple'},
                        {(13, 40): 'carrying a wine glass'},
                        {(26, 48): 'pointing a sandwich'},
                        {(26, 62): 'pointing a tv'},
                        {(13, 74): 'carrying a clock'},
                        {(5, 61): 'looking at a toilet'},
                        {(26, 19): 'pointing a cow'},
                        {(5, 65): 'looking at a remote'},
                        {(26, 18): 'pointing a sheep'},
                        {(0, 50): 'holding a broccoli'},
                        {(0, 13): 'holding a bench'},
                        {(26, 33): 'pointing a kite'},
                        {(0, 7): 'holding a truck'},
                        {(13, 41): 'carrying a cup'},
                        {(24, 45): 'drinking a bowl'},
                        {(13, 38): 'carrying a tennis racket'},
                        {(13, 39): 'carrying a bottle'},
                        {(5, 47): 'looking at an apple'},
                        {(5, 56): 'looking at a chair'},
                        {(2, 24): 'sitting a backpack'},
                        {(26, 60): 'pointing a dining table'},
                        {(0, 78): 'holding a hair drier'},
                        {(5, 39): 'looking at a bottle'},
                        {(26, 55): 'pointing a cake'},
                        {(26, 66): 'pointing a keyboard'},
                        {(26, 72): 'pointing a refrigerator'},
                        {(5, 74): 'looking at a clock'},
                        {(0, 8): 'holding a boat'},
                        {(17, 45): 'cutting a bowl'},
                        {(26, 23): 'pointing a giraffe'},
                        {(5, 25): 'looking at a umbrella'},
                        {(0, 66): 'holding a keyboard'},
                        {(2, 26): 'sitting a handbag'},
                        {(26, 52): 'pointing a hot dog'},
                        {(2, 60): 'sitting a dining table'},
                        {(13, 77): 'carrying a teddy bear'},
                        {(0, 51): 'holding a carrot'},
                        {(13, 34): 'carrying a baseball bat'},
                        {(5, 2): 'looking at a car'}, {(3, 5): 'riding a bus'},
                        {(17, 50): 'cutting a broccoli'},
                        {(5, 14): 'looking at a bird'},
                        {(13, 73): 'carrying a book'},
                        {(5, 50): 'looking at a broccoli'}]
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def calculate_iou(box1, box2):
    # 提取每个边界框的坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集矩形的坐标
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # 计算交集的面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算每个边界框的面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算联合面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou