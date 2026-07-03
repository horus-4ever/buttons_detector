from torch import nn, Tensor
import torch.nn.functional as F
import torch
from utils import FFN
from typing import Optional
from attention import MultiscaleDeformableAttention


class Encoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            nlevels: int,
            nlayers: int,
            npoints: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # module of nlayers encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                nheads=nheads,
                nlevels=nlevels,
                npoints=npoints,
                dim_ffn=dim_ffn,
                dropout=dropout,
                activation=activation
            )
            for _ in range(nlayers)
        ])
        # normalization layer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, spatial_shapes, pos_embed: Optional[Tensor], src_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [batch, sum_l(Hl~ * Wl~), embed_dim]
        - pos_embed: [batch, sum_l(Hl * Wl), embed_dim]
        - spatial_shapes: [num_levels, 2]
        """
        result = input
        encoder_attn_weights = []
        encoder_sampling_locations = []
        for layer in self.encoder_layers:
            result, attn_weights, sampling_locations = layer(result, spatial_shapes, pos_embed=pos_embed, src_key_padding_mask=src_key_padding_mask)
            encoder_attn_weights.append(attn_weights)
            encoder_sampling_locations.append(sampling_locations)
        # normalize
        result = self.norm(result)
        return result, encoder_attn_weights, encoder_sampling_locations


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            nlevels: int,
            npoints: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # multihead self-attention module
        self.multiscale_deformable_attention = MultiscaleDeformableAttention(
            embed_dim=d_model,
            num_heads=nheads,
            num_levels=nlevels,
            num_points=npoints,
        )
        # feed-forward network
        self.ffn = FFN(d_model, dim_ffn, dropout, activation=activation)
        # normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _get_reference_points(self, spatial_shapes, device, dtype):
        """
        - spatial_shapes: [num_levels, 2]

        - output: [sum_l(Hl * Wl), 2]
        
        Reference points are 2D points normalized between 0 and 1.
        In the encoder, we set one reference point per pixel of each feature map.
        """
        # now loop over the layers
        reference_points = []
        for _, (H, W) in enumerate(spatial_shapes.tolist()):
            # now we create the points for this layer using a meshgrid
            # first, define with linspace the coordonates
            width_spans = torch.linspace((0.5 / W), (1.0 - 0.5 / W), steps=W, device=device, dtype=dtype)
            height_spans = torch.linspace((0.5 / H), (1.0 - 0.5 / H), steps=H, device=device, dtype=dtype)
            i_indices, j_indices = torch.meshgrid(height_spans, width_spans)
            # i_indices, j_indices: [H, W]
            # now combine that to obtain the reference points
            points = torch.stack([i_indices, j_indices]) # [2, H, W]
            # [2, H, W] -> [H, W, 2]
            points = points.permute(1, 2, 0).contiguous()
            # [H * W, 2]
            points = points.view(H * W, 2)
            reference_points.append(points)
        output = torch.cat(reference_points, dim=0) # [sum_l(Hl * Wl), 2]
        return output

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, input, spatial_shapes, pos_embed: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [batch, sum_l(Hl~ * Wl~), embed_dim]
        - spatial_shapes: [num_levels, 2]
        - pos_embed: [batch, sum_l(Hl * Wl), embed_dim]
        - src_key_padding_mask: [B, 1, sum_l(Hl, Wl)]
        """
        B, Q, C = input.size()
        # compute Q and K matrices and apply positional embedding to it
        query = self.with_pos_embed(input, pos_embed)
        # get the reference points
        # [sum_l(Wl * Hl), 2]
        reference_points = self._get_reference_points(spatial_shapes, device=input.device, dtype=input.dtype)
        # convert the reference points for the multiscale attention
        # [sum_l(Wl * Hl), 2] -> [B, sum_l(Wl * Hl), num_levels, 2]
        reference_points = reference_points.unsqueeze(0).unsqueeze(2).expand(B, -1, spatial_shapes.shape[0], 2)
        # compute self-attention and dropout
        # query: [batch, sum_l(Hl~ * Wl~), embed_dim]
        self_att_out, attn_weights, sampling_locations = self.multiscale_deformable_attention(
            reference_points=reference_points, # [B, sum_l(Wl * Hl), num_levels, 2]
            spatial_shapes=spatial_shapes, # [num_levels, 2]
            query=query, # [batch, query_len, embed_dim]
            values=input, # [batch, sum_l(Hl~ * Wl~), embed_dim]
            key_padding_mask=src_key_padding_mask
        )
        # [B, query_len, embed_dim]
        self_att_out = self.dropout1(self_att_out)
        # first add and normalize
        add_norm_out = input + self_att_out
        add_norm_out = self.norm1(add_norm_out)
        # feed into the FFN and dropout
        ffn_out = self.ffn(add_norm_out)
        ffn_out = self.dropout2(ffn_out)
        # second add and normalize
        result = add_norm_out + ffn_out
        result = self.norm2(result)
        return result, attn_weights, sampling_locations
