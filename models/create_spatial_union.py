import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import torchvision.ops.boxes as box_ops
from util.box_ops import box_cxcywh_to_xyxy


def compute_box_pe_no_embed(boxes, image_size):
    bx_norm = boxes / image_size[[1, 0, 1, 0]]
    bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
    b_wh = bx_norm[:, 2:] - bx_norm[:, :2]
    c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)  # (30,1,256)
    wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)
    box_pe = torch.cat([c_pe, wh_pe], dim=-1)
    return box_pe


def generate_ho_coord(h_o_coord_cat):
    reference_points_input_x = (h_o_coord_cat[:, :, 0] + h_o_coord_cat[:, :, 4])/2
    reference_points_input_y = (h_o_coord_cat[:, :, 1] + h_o_coord_cat[:, :, 5])/2
    reference_points_input_w = torch.abs(h_o_coord_cat[:, :, 0] - h_o_coord_cat[:, :, 4]) \
        + (h_o_coord_cat[:, :, 2] + h_o_coord_cat[:, :, 6])/2
    reference_points_input_h = torch.abs(h_o_coord_cat[:, :, 1] - h_o_coord_cat[:, :, 5]) \
        + (h_o_coord_cat[:, :, 3] + h_o_coord_cat[:, :, 7])/2
    ho_coord = torch.stack([reference_points_input_x, reference_points_input_y, reference_points_input_w, reference_points_input_h],-1)
    return ho_coord


def compute_sinusoidal_pe(pos_tensor: Tensor, temperature: float = 10000.) -> Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)


def preprocess_boxes(h_boxes,o_boxes,ho_boxes,image_sizes):
    # convert to [x0, y0, x1, y1] format
    h_boxes = box_cxcywh_to_xyxy(h_boxes)
    o_boxes = box_cxcywh_to_xyxy(o_boxes)
    ho_boxes = box_cxcywh_to_xyxy(ho_boxes)
    img_h, img_w = image_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    h_boxes = h_boxes * scale_fct[:, None, :]
    o_boxes = o_boxes * scale_fct[:, None, :]
    ho_boxes = ho_boxes * scale_fct[:, None, :]
    return h_boxes, o_boxes, ho_boxes


class SpatialCreator(nn.Module):
    def __init__(self, dim=256, use_spatial_pos=False, use_spatial_emd=False):
        super().__init__()
        self.use_spatial_pos = use_spatial_pos
        self.use_spatial_emd = use_spatial_emd
        # if self.use_spatial_pos:
        #     self.ref_anchor_head = nn.Sequential(
        #         nn.Linear(256, 256), nn.ReLU(),
        #         nn.Linear(256, 2))
        if use_spatial_emd:
            self.spatial_head = nn.Sequential(
                nn.Linear(36, 128), nn.ReLU(),
                nn.Linear(128, dim), nn.ReLU()
                #nn.Linear(128, 256), nn.ReLU(),
                #nn.Linear(256, dim),
            )

    def compute_box_pe(self, boxes, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]
        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)  # (30,1,256)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)
        box_pe = torch.cat([c_pe, wh_pe], dim=-1)
        # ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()  # n_query, 2
        # # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        # c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        # c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)
        return box_pe




    def process_pair(self, h_boxes, o_boxes, image_sizes):
        data_list = []
        bs = h_boxes.shape[0]
        h_boxes, o_boxes = preprocess_boxes(h_boxes, o_boxes, image_sizes)
        for i in range(bs):
            data_dict = {
                'h_boxes': h_boxes[i],
                'o_boxes': o_boxes[i]
            }
            data_list.append(data_dict)

        for i, data in enumerate(data_list):
            h_boxes, o_boxes = data.values()
            pairwise_spatial = compute_spatial_encodings(
                [h_boxes, ], [o_boxes, ], [image_sizes[i], ]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)


            pairwise_spatial = torch.unsqueeze(pairwise_spatial, dim=0)
            if i == 0:
                paired_spatial = pairwise_spatial
            else:
                paired_spatial = torch.cat((paired_spatial, pairwise_spatial), dim=0)
        paired_spatial = paired_spatial.transpose(0, 1)

        return paired_spatial


    def process(self, h_boxes, o_boxes, ho_boxes, image_sizes):
        data_list = []
        bs = h_boxes.shape[0]
        h_boxes, o_boxes, ho_boxes = preprocess_boxes(h_boxes, o_boxes, ho_boxes, image_sizes)
        for i in range(bs):
            data_dict = {
                'h_boxes': h_boxes[i],
                'o_boxes': o_boxes[i],
                'ho_boxes': ho_boxes[i],
            }
            data_list.append(data_dict)
        h_position_emd = {}
        o_position_emd = {}
        ho_position_emd = {}

        for i, data in enumerate(data_list):
            h_boxes, o_boxes, ho_boxes = data.values()
            pairwise_spatial = compute_spatial_encodings(
                [h_boxes, ], [o_boxes, ], [image_sizes[i], ]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            # pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)
            h_box_pe = self.compute_box_pe(h_boxes, image_sizes[i])
            o_box_pe = self.compute_box_pe(o_boxes, image_sizes[i])
            ho_box_pe = compute_box_pe_no_embed(ho_boxes, image_sizes[i])
            h_box_pe = torch.unsqueeze(h_box_pe, dim=0)
            o_box_pe = torch.unsqueeze(o_box_pe, dim=0)
            ho_box_pe = torch.unsqueeze(ho_box_pe, dim=0)
            # h_c_pe = torch.unsqueeze(h_c_pe, dim=0)
            # o_c_pe = torch.unsqueeze(o_c_pe, dim=0)
            if i == 0:
                h_boxes_pe = h_box_pe
                o_boxes_pe = o_box_pe
                ho_boxes_pe = ho_box_pe
                # h_ces_pe = h_c_pe
                # o_ces_pe = o_c_pe
            else:
                h_boxes_pe = torch.cat((h_boxes_pe, h_box_pe), dim=0)
                o_boxes_pe = torch.cat((o_boxes_pe, o_box_pe), dim=0)
                ho_boxes_pe = torch.cat((ho_boxes_pe, ho_box_pe), dim=0)
                # h_ces_pe = torch.cat((h_ces_pe, h_c_pe), dim=0)
                # o_ces_pe = torch.cat((o_ces_pe, o_c_pe), dim=0)

            pairwise_spatial = torch.unsqueeze(pairwise_spatial, dim=0)
            if i == 0:
                paired_spatial = pairwise_spatial
            else:
                paired_spatial = torch.cat((paired_spatial, pairwise_spatial), dim=0)
        paired_spatial = paired_spatial.transpose(0, 1)
        # h_position_emd["centre"] = h_ces_pe.transpose(0, 1)
        h_position_emd["box"] = h_boxes_pe.transpose(0, 1)
        ho_position_emd["box"] = ho_boxes_pe.transpose(0, 1)
        # o_position_emd["centre"] = o_ces_pe.transpose(0, 1)
        o_position_emd["box"] = o_boxes_pe.transpose(0, 1)
        return h_position_emd, o_position_emd, ho_position_emd, paired_spatial



    def forward(self, h_boxes, o_boxes, image_sizes, device=None):
        # h_boxes(bs,64,4);h_f(bs,64,256)
        if device == None:
            device = h_boxes[0].device

        if self.use_spatial_pos and self.use_spatial_emd:
            if len(h_boxes.shape) == 4:
                h_intermediate = []
                o_intermediate = []
                ho_intermediate = []
                pair_intermediate = []

                for i in range(h_boxes.shape[0]):
                    ho_boxes = generate_ho_coord(torch.cat((h_boxes[i], o_boxes[i]), dim=-1))
                    h_pos, o_pos, ho_pos, pair_embed = self.process(h_boxes[i], o_boxes[i], ho_boxes, image_sizes)
                    h_intermediate.append(h_pos)
                    o_intermediate.append(o_pos)
                    ho_intermediate.append(ho_pos)
                    pair_intermediate.append(pair_embed)
                pair_embed = torch.stack(pair_intermediate, dim=0)
            else:
                ho_boxes = generate_ho_coord(torch.cat((h_boxes, o_boxes), dim=-1))
                h_intermediate, o_intermediate, ho_intermediate, pair_embed = self.process(h_boxes, o_boxes, ho_boxes,
                                                                                           image_sizes)

            return h_intermediate, o_intermediate,ho_intermediate, pair_embed
        elif self.use_spatial_emd:
            if len(h_boxes.shape) == 4:
                pair_intermediate = []

                for i in range(h_boxes.shape[0]):
                    pair_embed = self.process_pair(h_boxes[i], o_boxes[i], image_sizes)

                    pair_intermediate.append(pair_embed)
                pair_embed = torch.stack(pair_intermediate, dim=0)
            else:
                pair_embed = self.process_pair(h_boxes, o_boxes, image_sizes)

            return None, None, None, pair_embed
        elif self.use_spatial_pos:
            if len(h_boxes.shape) == 4:
                h_intermediate = []
                o_intermediate = []
                ho_intermediate = []

                for i in range(h_boxes.shape[0]):
                    ho_boxes = generate_ho_coord(torch.cat((h_boxes[i], o_boxes[i]), dim=-1))
                    h_pos, o_pos, ho_pos, _ = self.process(h_boxes[i], o_boxes[i], ho_boxes, image_sizes)
                    h_intermediate.append(h_pos)
                    o_intermediate.append(o_pos)
                    ho_intermediate.append(ho_pos)
            else:
                ho_boxes = generate_ho_coord(torch.cat((h_boxes, o_boxes), dim=-1))
                h_intermediate, o_intermediate, ho_intermediate, _ = self.process(h_boxes, o_boxes, ho_boxes,
                                                                                           image_sizes)

            return h_intermediate, o_intermediate,ho_intermediate, None
        else:
            return None, None, None, None





# if __name__ == '__main__':
#     h_boxes = torch.rand((3, 2, 64, 4))
#     o_boxes = torch.rand((3, 2, 64, 4))
#     h_f = torch.rand((3, 2, 64, 256))
#     o_f = torch.rand((3, 2, 64, 256))
#     image_sizes = torch.tensor([[800, 1207], [800, 1207]])
#     model = SpatialCreator()
#     a, b, c = model(h_boxes, o_boxes, h_f, o_f, image_sizes)
#     print(1)
def build_SpatialCreator(args):
    return SpatialCreator(use_spatial_pos=args.use_spatial_pos,
                          use_spatial_emd=args.use_spatial_emd)



