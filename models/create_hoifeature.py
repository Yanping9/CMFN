import torch
import torch.nn as nn
from gitdb.db import mem
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Optional, List
import copy
from .attention import MultiheadAttention


class HoiFeatureCreator(nn.Module):
    def __init__(self, d_model=512, clip_dim=768, nhead=8,
                 num_dec_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True,
                 return_intermediate_dec=False, use_spa_pos=False,use_spa_emd=False, use_clip_aux=False):
        super().__init__()
        decoder_layer = FelaDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                         normalize_before, use_spa_pos=use_spa_pos)
        decoder_norm = nn.LayerNorm(d_model)
        hoi_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        hoi_decoder_norm = nn.LayerNorm(d_model)

        self.decoder = FelaDecoder(decoder_layer, 1, decoder_norm, return_intermediate=False)
        self.hoi_decoder = TransformerDecoder(hoi_decoder_layer, num_dec_layers,
                                              hoi_decoder_norm, return_intermediate_dec)
        if use_clip_aux:
            self.pooling = nn.AvgPool1d(kernel_size=3, stride=3)
        else:
            mem_ecoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                                       normalize_before=normalize_before)
            encoder_norm = nn.LayerNorm(d_model)
            self.inswise_mem = TransformerEncoder(mem_ecoder_layer, 1, encoder_norm)
        self.ho_ln = nn.Sequential(
            nn.Linear(256, 256)
        )
        if use_spa_emd:
            self.fusion_ln = nn.Sequential(
                nn.Linear(512, 256)
            )
        self.ATF = ATFmodule()

    def spatial_fusion(self, x: Tensor, y: Tensor):
        fusion_ho = torch.cat((x, y), dim=-1)
        fusion_ho = self.fusion_ln(fusion_ho)
        return fusion_ho


    def fela(self, h_f, o_f, mem, paired_emd=None, h_pos=None, o_pos=None, ho_pos=None, mem_pos=None, mem_mask=None):
        post_h_f, post_o_f = self.decoder(h_f, o_f, mem, h_query_pos=h_pos, o_query_pos=o_pos,
                                          union_pos=ho_pos, pos=mem_pos, memory_key_padding_mask=mem_mask)
        inter_hs = (post_h_f + post_o_f) / 2.0
        inter_hs = self.ho_ln(inter_hs)
        if paired_emd != None:
            inter_hs = self.spatial_fusion(inter_hs, paired_emd)
        origin_hs = (h_f + o_f) / 2.0
        inter_hs = self.ATF(origin_hs, inter_hs)
        # inter_hs = self.ATF(inter_hs, origin_hs)
        return inter_hs

    def forward(self, h_f, o_f, mask, memory, mem_pos_emd, paired_emd=None,
                clip_src=None, h_pos=None, o_pos=None, ho_pos=None):
        if clip_src is not None:
            ins_context = self.pooling(clip_src)
            ic_pos = None
            ic_mask = None
        else:
            ins_context = self.inswise_mem(memory, pos=mem_pos_emd, src_key_padding_mask=mask)  # (672,bs,256)
            ic_pos = mem_pos_emd
            ic_mask = mask

        if len(h_f.shape) == 4:
            intermediate = []
            for i in range(h_f.shape[0]):
                if paired_emd is None:
                    if h_pos is None:
                        output = self.fela(h_f[i], o_f[i], ins_context, mem_pos=ic_pos, mem_mask=ic_mask)
                    else:
                        output = self.fela(h_f[i], o_f[i], ins_context, h_pos=h_pos, o_pos=o_pos,
                                           ho_pos=ho_pos, mem_pos=ic_pos, mem_mask=ic_mask)
                elif len(paired_emd.shape) == 4:
                    if h_pos is not None:
                        output = self.fela(h_f[i], o_f[i], ins_context, paired_emd[i],
                                           h_pos[i], o_pos[i], ho_pos[i], ic_pos, ic_mask)
                    else:
                        output = self.fela(h_f[i], o_f[i], ins_context, paired_emd[i], mem_pos=ic_pos, mem_mask=ic_mask)
                else:
                    if h_pos is not None:
                        output = self.fela(h_f[i], o_f[i], ins_context, paired_emd, h_pos, o_pos, ho_pos, ic_pos, ic_mask)
                    else:
                        output = self.fela(h_f[i], o_f[i], ins_context, paired_emd, mem_pos=ic_pos, mem_mask=ic_mask)

                intermediate.append(output)
            inter_hs = torch.stack(intermediate)
            inter_tgt = torch.zeros_like(inter_hs[0])
        else:
            if len(paired_emd.shape) == 3:
                if h_pos is not None:
                    inter_hs = self.fela(h_f, o_f, ins_context, paired_emd, h_pos, o_pos, ho_pos, ic_pos, ic_mask)
                else:
                    inter_hs = self.fela(h_f, o_f, ins_context, paired_emd, mem_pos=ic_pos, mem_mask=ic_mask)

                inter_tgt = torch.zeros_like(inter_hs)
            else:
                intermediate = []
                for i in range(paired_emd.shape[0]):
                    if h_pos is not None:
                        output = self.fela(h_f, o_f, ins_context, paired_emd[i],
                                           h_pos[i], o_pos[i], ho_pos[i], ic_pos, ic_mask)
                    else:
                        output = self.fela(h_f, o_f, ins_context, paired_emd[i], mem_pos=ic_pos, mem_mask=ic_mask)
                    intermediate.append(output)
                inter_hs = torch.stack(intermediate)
                inter_tgt = torch.zeros_like(inter_hs[0])

        inter_hs = self.hoi_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
                                    pos=mem_pos_emd, query_pos=inter_hs)
        inter_hs = inter_hs.transpose(1, 2)
        return inter_hs, ins_context


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class FelaDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt_h, tgt_o, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                h_query_pos: Optional[Tensor] = None,
                o_query_pos: Optional[Tensor] = None,
                union_pos: Optional[Tensor] = None):
        output_h = tgt_h
        output_o = tgt_o


        for i, layer in enumerate(self.layers):

            # this_query_pos = query_pos
            output_h, output_o = layer(output_h, output_o, memory, tgt_mask=tgt_mask,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, h_query_pos=h_query_pos,
                                       o_query_pos=o_query_pos,
                                       union_pos=union_pos)


        return output_h, output_o


class FelaDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_spa_pos=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model*2, nhead, dropout=dropout)
        self.multihead_attn_h = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_o = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear_ho1 = nn.Linear(d_model*2, dim_feedforward // 2)
        self.dropout_ho1 = nn.Dropout(dropout)
        self.dropout_ho2 = nn.Dropout(dropout)
        self.linear_ho2 = nn.Linear(dim_feedforward // 2, d_model*2)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_h = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout_o = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm_ho1 = nn.LayerNorm(d_model*2)
        self.norm_ho2 = nn.LayerNorm(d_model*2)
        self.norm = nn.LayerNorm(d_model*2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        if use_spa_pos:
            self.h_pos_proj = nn.Linear(512, 256)
            self.o_pos_proj = nn.Linear(512, 256)
            self.union_pos_proj = nn.Linear(512, 256)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_pre(self, tgt_h, tgt_o, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    h_query_pos: Optional[Tensor] = None,
                    o_query_pos: Optional[Tensor] = None,
                    union_query_pos: Optional[Tensor] = None
                    ):
        tgt = torch.cat((tgt_h, tgt_o), dim=-1)
        if h_query_pos is not None:
            h_pos = self.h_pos_proj(h_query_pos["box"])
            o_pos = self.o_pos_proj(o_query_pos["box"])
            union_pos = self.union_pos_proj(union_query_pos['box'])
            qkv_pos = torch.cat([h_pos, o_pos], dim=-1)
            q_h_pos = h_pos + union_pos
            q_o_pos = o_pos + union_pos

        else:
            qkv_pos = None
            q_h_pos = None
            q_o_pos = None
        tgt2 = self.norm(tgt)

        q = k = self.with_pos_embed(tgt2, qkv_pos)

        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm_ho1(tgt)
        tgt2 = self.linear_ho2(self.dropout_ho1(self.activation(self.linear_ho1(tgt2))))
        tgt = tgt + self.dropout_ho2(tgt2)
        tgt = self.norm_ho2(tgt)
        tgt_h = tgt_h + tgt[:, :, :256]
        tgt_o = tgt_o + tgt[:, :, 256:]
        tgt_h1 = self.norm1(tgt_h)
        tgt_o1 = self.norm2(tgt_o)
        # hum_branch
        tgt_h1 = self.multihead_attn_h(query=self.with_pos_embed(tgt_h1, q_h_pos),
                                      key=self.with_pos_embed(memory, pos),
                                      value=memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)[0]
        tgt_h = tgt_h + self.dropout2(tgt_h1)
        tgt_h2 = self.norm3(tgt_h)
        tgt_h2 = self.linear2(self.dropout_h(self.activation(self.linear1(tgt_h2))))
        tgt_h = tgt_h + self.dropout3(tgt_h2)

        # obj_branch
        tgt_o1 = self.multihead_attn_o(query=self.with_pos_embed(tgt_o1, q_o_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt_o = tgt_o + self.dropout2(tgt_o1)
        tgt_o2 = self.norm4(tgt_o)
        tgt_o2 = self.linear4(self.dropout_o(self.activation(self.linear3(tgt_o2))))
        tgt_o = tgt_o + self.dropout3(tgt_o2)
        return tgt_h, tgt_o

    def forward(self, tgt_h,tgt_o, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                h_query_pos: Optional[Tensor] = None,
                o_query_pos: Optional[Tensor] = None,
                union_pos: Optional[Tensor] = None):

        return self.forward_pre(tgt_h, tgt_o, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, h_query_pos, o_query_pos, union_pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class ATFmodule(nn.Module):
    def __init__(self):
        super(ATFmodule, self).__init__()
        self.attetion_block = nn.Sequential(nn.Linear(256*2,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.Sigmoid())
        self.mlp_layer = nn.Sequential(nn.Linear(256*2,256),
                                       nn.ReLU(),
                                       nn.Linear(256,256),
                                       nn.LayerNorm(256))

    def forward(self, x, y):
        att = self.attetion_block(torch.cat([x, y], dim=-1))  # att:cat->(nq,bs,512)->atten_block(mlp)->(nq,bs,256)
        ret = x + att * self.mlp_layer(torch.cat([x, y], dim=-1))  # ret(nq,bs,256)
        return ret


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_hoi_feature(args):
    return HoiFeatureCreator(d_model=args.hidden_dim,
                             clip_dim=args.clip_dim,
                             dropout=args.dropout,
                             nhead=args.nheads,
                             dim_feedforward=args.dim_feedforward,
                             num_dec_layers=args.dec_layers,
                             normalize_before=True,
                             return_intermediate_dec=True,
                             use_spa_pos=args.use_spatial_pos,
                             use_spa_emd=args.use_spatial_emd,
                             use_clip_aux=args.with_clip_input)