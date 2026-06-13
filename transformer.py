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

    def forward(self, features, query_embed, pos_embeds, masks):
        """
        - features: num_levels * [B, embed_dim, Hl, Wl]
        - query_embed: [num_queries, embed_dim]
        - pos_embeds: num_levels * [B, embed_dim, Hl, Wl]
        - masks: num_levels * [B, 1, Hl, Wl]
        """
        # prepare input for encoder
        feat_flatten = []
        mask_flatten = []
        pos_flatten = []
        spatial_shapes = []
        for level, (feature_map, mask, positions) in enumerate(zip(features, masks, pos_embeds)):
            B, Q, H, W = feature_map.size()
            spatial_shapes.append((H, W))
            # [B, embed_dim, Hl, Wl] -> [B, embed_dim, Hl * Wl]
            feature_map = feature_map.view(B, -1, H * W)
            feat_flatten.append(feature_map)
            # [B, embed_dim, Hl, Wl] -> [B, embed_dim, Hl * Wl]
            positions = positions.view(B, -1, H * W)
            pos_flatten.append(positions)
            # [B, 1, Hl, Wl] -> [B, 1, Hl * Wl]
            mask = mask.view(B, -1, H * W)
            mask_flatten.append(mask)
        # [B, embed_dim, sum_l(Hl, Wl)]
        feat_flatten = torch.cat(feat_flatten, dim=2)
        mask_flatten = torch.cat(mask_flatten, dim=2)
        pos_flatten = torch.cat(pos_flatten, dim=2)
        
        # decoder input
        decoder_input = torch.zeros_like(query_embed)
        # forward of encoder
        memory = self.encoder(f, p, m)
        # forward of decoder
        # result of shape [B, num_queries, C]
        result, attn_maps = self.decoder(decoder_input, memory, pos=pos_embed, queries_pos=query_embed, att_map_size=(att_height, att_width), memory_key_padding_mask=mask)
        return result, memory, attn_maps
