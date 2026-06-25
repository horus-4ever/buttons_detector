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
            nlevels: int,
            npoints: int,
            nrefpointsperquery: int,
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
            nlevels=nlevels,
            npoints=npoints,
            nlayers=encoder_nlayers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        # decoder
        self.decoder = Decoder(
            d_model=d_model,
            nheads=nheads,
            nlevels=nlevels,
            npoints=npoints,
            nrefpointsperquery=nrefpointsperquery,
            nlayers=decoder_nlayers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        # level embedding is learned
        self.level_embed = nn.Parameter(torch.Tensor(nlevels, d_model))
        nn.init.normal_(self.level_embed) # initializa the values
        # learned reference points
        # reference points in the decoder are learned from linear projection from object queries
        self.proj_reference_points = nn.Linear(d_model, 2)

    def forward(self, features, query_embed, refpoints_embed, pos_embeds, masks):
        """
        - features: num_levels * [B, embed_dim, Hl, Wl]
        - query_embed: [num_queries, embed_dim]
        - refpoints_embed: [RpQ, embed_dim]
        - pos_embeds: num_levels * [B, embed_dim, Hl, Wl]
        - masks: num_levels * [B, 1, Hl, Wl]
        """
        # prepare input for encoder
        feat_flatten = []
        mask_flatten = []
        pos_flatten = []
        spatial_shapes = []
        for level, (feature_map, mask, positions) in enumerate(zip(features, masks, pos_embeds)):
            B, _, H, W = feature_map.size()
            spatial_shapes.append((H, W))
            # [B, embed_dim, Hl, Wl] -> [B, embed_dim, Hl * Wl]
            feature_map = feature_map.view(B, -1, H * W)
            feat_flatten.append(feature_map)
            # level embedding, add them to the position embedding
            # [embed_dim] -> [1, 1, embed_dim]
            level_embedding = self.level_embed[level].view(1, 1, -1)
            # [B, embed_dim, Hl, Wl] -> [B, embed_dim, Hl * Wl]
            positions = positions.view(B, -1, H * W)
            # [B, embed_dim, Hl * Wl] -> [B, Hl * Wl, embed_dim]
            positions = positions.permute(0, 2, 1).contiguous()
            pos_flatten.append(positions + level_embedding)
            # [B, 1, Hl, Wl] -> [B, 1, Hl * Wl]
            mask = mask.view(B, -1, H * W)
            mask_flatten.append(mask)
        # [B, suml(Hl * Wl), embed_dim]
        feat_flatten = torch.cat(feat_flatten, dim=2).permute(0, 2, 1).contiguous()
        # [B, 1, suml(Hl * Wl)]
        mask_flatten = torch.cat(mask_flatten, dim=2)
        # [B, suml(Hl * Wl), embed_dim]
        pos_flatten = torch.cat(pos_flatten, dim=1)
        # make the spatial shapes be a tensor
        spatial_shapes = torch.tensor(spatial_shapes, device=query_embed.device)
        # forward of encoder
        memory, encoder_attention_weights, encoder_sampling_locations = self.encoder(
            input=feat_flatten, # [batch, sum_l(Hl * Wl), embed_dim]
            spatial_shapes=spatial_shapes, # [num_levels, 2]
            pos_embed=pos_flatten, # [B, embed_dim, sum_l(Hl, Wl)]
            src_key_padding_mask=mask_flatten, # [B, 1, sum_l(Hl, Wl)]
        ) # [B, sum_l(Hl * Wl), embed_dim]

        B, Q, C = memory.size()
        # now prepare the input to the decoder
        # [num_queries, embed_dim]
        object_queries = torch.zeros_like(query_embed)
        # [num_queries, embed_dim] -> [B, num_queries, embed_dim]
        object_queries = object_queries.expand(B, -1, -1)
        # now we get the reference points by linear projection
        # [Q, C] -> [Q, 1, C]
        query_tokens = query_embed[:, None, :]
        # [RqP, C] -> [1, RqP, C]
        role_tokens = refpoints_embed[None, :, :]
        # [Q, RqP, C]
        token_embed = query_tokens + role_tokens # we construct from both query and reference points embedings
        # [Q, RqP, 2]
        reference_points = self.proj_reference_points(token_embed).sigmoid()
        result, decoder_attn_maps, decoder_sampling_locations = self.decoder(
            input=object_queries, # [B, num_queries, embed_dim]
            memory=memory, # [B, sum_l(Hl * Wl), embed_dim]
            reference_points=reference_points, # [query_len, RpQ, 2]
            spatial_shapes=spatial_shapes, # [num_levels, 2]
            query_embed=query_embed, # [num_queries, embed_dim]
            refpoints_embed=refpoints_embed, # [RpQ, embed_dim]
            memory_key_padding_mask=mask_flatten, # [B, 1, suml(Hl * Wl)]
        )
        # decoder_attn_weights: decoder_layers * [batch, query_len * RqP, heads, num_levels, num_points]
        # result: [B, Q, RpQ, embed_dim]
        return result, decoder_attn_maps, spatial_shapes, decoder_sampling_locations, reference_points
