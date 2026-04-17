from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
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

    def forward(self, src, query_embed, pos_embed, mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, att_height, att_width = pos_embed.shape
        src = src.flatten(2).transpose(1, 2)          # [B, HW, C]
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, HW, C]
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)  # [B, num_queries, C]
        mask = mask.flatten(1)  # [B, HW]
        # decoder input
        decoder_input = torch.zeros_like(query_embed)
        # forward of encoder
        memory = self.encoder(src, pos=pos_embed, src_key_padding_mask=mask)
        # forward of decoder
        result = self.decoder(decoder_input, memory, pos=pos_embed, queries_pos=query_embed, att_map_size=(att_height, att_width), memory_key_padding_mask=mask)
        return result, memory
