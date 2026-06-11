from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from encoder import Encoder
from decoder import Decoder


class DeformableTransformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            encoder_nlayers: int,
            decoder_nlayers: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # encoder
        self.encoder = Encoder(
            d_model=d_model,
            nheads=nheads,
            nlayers=encoder_nlayers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        # decoder
        self.decoder = Decoder(
            d_model=d_model,
            nheads=nheads,
            nlayers=decoder_nlayers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )

    def forward(self, feat, query_embed, pos_embed, masks):
        """
        - feat: [B, 256, H~, W~]
        - query_embed: [num_queries, 256]
        - pos_embed: [B, 256, H~, W~]
        - masks: [B, 1, H~, W~]
        """
        # prepare input for encoder
        feat_flatten = []
        mask_flatten = []
        pos_flatten = []
        for src, mask, pos in zip(feat, masks, pos_embed):
            # TODO: flatten
            pass
        src_flatten = torch.stack(feat_flatten)
        mask_flatten = torch.stack(mask_flatten)
        pos_flatten = torch.stack(pos_flatten)
        # decoder input
        decoder_input = torch.zeros_like(query_embed)
        # forward of encoder
        memory = self.encoder(feat_flatten, pos_embed, masks)
        # forward of decoder
        # result of shape [B, num_queries, C]
        result, attn_maps = self.decoder(decoder_input, memory, pos=pos_embed, queries_pos=query_embed, att_map_size=(att_height, att_width), memory_key_padding_mask=mask)
        return result, memory, attn_maps
