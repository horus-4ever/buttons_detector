from torch import nn, Tensor
import torch.nn.functional as F
from utils import FFN
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

    def forward(self, input, memory, reference_points, spatial_shapes, pos: Optional[Tensor], queries_pos: Optional[Tensor], memory_key_padding_mask: Optional[Tensor] = None):
        B, num_queries, _ = input.shape
        attn_maps = []
        # loop over the decoder layers
        output = input
        for i, layer in enumerate(self.layers):
            output, weights, sampling_locations = layer(
                input=output, 
                memory=memory, 
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                pos=pos,
                queries_pos=queries_pos,
                memory_key_padding_mask=memory_key_padding_mask
            )
            # TODO: redo the forward of this
        # normalize and return
        output = self.norm(output)
        return output, attn_maps


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

    def forward(self, input, memory, reference_points, spatial_shapes, pos: Optional[Tensor], queries_pos: Optional[Tensor], memory_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [B, num_queries, embed_dim]
        - memory: [B, query_len, embed_dim]
        - pos: [B, num_queries, embed_dim]
        """
        # computes k and q for queries attention
        k_queries = q_queries = self.with_pos_embed(input, queries_pos)
        # compute self-attention on queries and dropout
        queries_attention_out = self.queries_attention(q_queries, k_queries, input)[0]
        queries_attention_out = self.dropout1(queries_attention_out)
        # add and normalize
        add_norm_out = input + queries_attention_out
        add_norm_out = self.norm1(add_norm_out)
        # computes v, k and q for memory attention
        v_memory = memory
        q_memory = self.with_pos_embed(add_norm_out, queries_pos)
        # compute self-attention
        memory_attention_out, memory_attention_weights, memory_attention_sampling_locations = self.memory_attention(
            query=q_memory,
            reference_points=reference_points,
            values=v_memory,
            spatial_shapes=spatial_shapes,
            key_padding_mask=memory_key_padding_mask
        )
        memory_attention_out = self.dropout2(memory_attention_out)
        # memory_attention_weights (average=False) --> [B, num_queries, source_size]
        # memory_attention_weights (average=True) --> [B, num_heads, num_queries, source_size]
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