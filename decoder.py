from torch import nn, Tensor
import torch.nn.functional as F
from utils import FFN, inverse_sigmoid
import torch
from typing import Optional
from attention import MultiscaleDeformableAttention


class Decoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            nlayers: int,
            nlevels: int,
            npoints: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
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
        # NEW: now we refine the reference points
        # we create one learnable refiner projection per decoder layer
        self.refpoints_refiners = nn.ModuleList(
            [nn.Linear(d_model, 2) for _ in range(nlayers)]
        )
        self._init_parameters()

    def _init_parameters(self):
        # initialize the refiners so that they begin at 0
        for refiner in self.refpoints_refiners:
            nn.init.zeros_(refiner.weight) # type: ignore
            nn.init.zeros_(refiner.bias) # type: ignore

    def forward(self, input, memory, reference_points, spatial_shapes, queries_pos: Optional[Tensor], memory_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [num_queries, embed_dim]
        - memory: [B, sum_l(Hl * Wl), embed_dim]
        - reference_points: [num_queries, 2]
        - spatial_shapes: [num_levels, 2]
        - pos: [B, suml(Hl * Wl), embed_dim]
        - queries_pos: [num_queries, embed_dim]
        - memory_key_padding_mask: [B, 1, suml(Hl * Wl)]
        """
        B, _, C = memory.size()
        decoder_attn_weights = []
        decoder_sampling_locations = []
        # reference points should be batch dependent so reshape them
        # [num_queries, 2] -> [B, num_queries, 2]
        reference_points = reference_points[None, :, :].expand(B, -1, -1).contiguous()
        intermediate_reference_points = []
        # loop over the decoder layers
        output = input
        for num_layer, layer in enumerate(self.layers):
            output, attn_weights, sampling_locations = layer(
                input=output,
                memory=memory,
                reference_points=reference_points, # [B, num_queries, 2]
                spatial_shapes=spatial_shapes,
                queries_pos=queries_pos,
                memory_key_padding_mask=memory_key_padding_mask
            )
            decoder_attn_weights.append(attn_weights)
            decoder_sampling_locations.append(sampling_locations)
            # refine the reference points for the decoder layers
            # -> [B, num_queries, RpQ, 2]
            ref_points_deltas = self.refpoints_refiners[num_layer](output)
            reference_points = (inverse_sigmoid(reference_points) + ref_points_deltas)
            reference_points = reference_points.sigmoid()
            # here, take the reference point and return it
            intermediate_reference_points.append(reference_points)
            # now detach it for next layer use
            reference_points = reference_points.detach()
        # decoder_attn_weights: decoder_layers * [batch, query_len, heads, num_levels, num_points]
        # normalize and return
        output = self.norm(output)
        return output, decoder_attn_weights, decoder_sampling_locations, intermediate_reference_points


class DecoderLayer(nn.Module):
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
        self.num_levels = nlevels
        # multihead self-attention layers
        self.queries_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True
        )
        # batch first --> [B, num_queries, C]
        self.memory_attention = MultiscaleDeformableAttention(
            embed_dim=d_model,
            num_heads=nheads,
            num_levels=nlevels,
            num_points=npoints,
        )
        # feed-forward layer
        self.ffn = FFN(d_model, dim_ffn, dropout=dropout, activation=activation)
        # add and norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, input, memory, reference_points, spatial_shapes, queries_pos: Optional[Tensor], memory_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [B, num_queries, embed_dim]
        - memory: [B, query_len, embed_dim]
        - reference_points: [B, query_len, 2]
        - queries_pos: [num_queries, embed_dim]
        - memory_key_padding_mask: 
        """
        B, Q, C = input.size()
        # computes k and q for queries attention
        k_queries = q_queries = self.with_pos_embed(input, queries_pos)
        # compute self-attention on queries and dropout
        queries_attention_out = self.queries_attention(q_queries, k_queries, input)[0]
        queries_attention_out = self.dropout1(queries_attention_out)
        # add and normalize
        add_norm_out = input + queries_attention_out
        add_norm_out = self.norm1(add_norm_out)
        # computes v, k and q for memory attention
        v_memory = memory # [B, query_len, embed_dim]
        # [num_queries, embed_dim]
        q_memory = self.with_pos_embed(add_norm_out, queries_pos)
        # [num_queries, embed_dim] -> [B, num_queries, embed_dim]
        q_memory = q_memory.expand(B, Q, C)
        # resize the reference_points to the right size
        # [batch, query_len, 2] -> [batch, query_len, num_levels, 2]
        reference_points = reference_points[:, :, None, :].expand(B, Q, self.num_levels, 2)
        # compute self-attention
        memory_attention_out, memory_attention_weights, memory_attention_sampling_locations = self.memory_attention(
            query=q_memory, # [B, num_queries, embed_dim]
            reference_points=reference_points, # [query_len, 2]
            values=v_memory, # [B, query_len, embed_dim]
            spatial_shapes=spatial_shapes, # [num_levels, 2]
            key_padding_mask=memory_key_padding_mask, # 
        )
        # memory_attention_weights: [batch, query_len, heads, num_levels, num_points]
        # [B, query_len, embed_dim]
        memory_attention_out = self.dropout2(memory_attention_out)
        # add and normalize
        add_norm_out = add_norm_out + memory_attention_out
        add_norm_out = self.norm2(add_norm_out)
        # ffn
        ffn_out = self.ffn(add_norm_out)
        ffn_out = self.dropout3(ffn_out)
        # add and normalize
        result = add_norm_out + ffn_out
        result = self.norm3(result)
        return result, memory_attention_weights, memory_attention_sampling_locations
