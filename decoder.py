from torch import nn, Tensor
import torch.nn.functional as F
from utils import FFN
import torch
from typing import Optional


class Decoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            nlayers: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                nheads=nheads,
                dim_ffn=dim_ffn,
                dropout=dropout,
                activation=activation
            )
            for _ in range(nlayers)
        ])
        # normalization layer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, memory, pos: Optional[Tensor], queries_pos: Optional[Tensor]):
        B, num_queries, _ = input.shape
        output = input
        self.attn_maps = []
        for layer in self.layers:
            output, weights = layer(output, memory, pos, queries_pos)
            # attn = weights[0][0]
            # memory_attention_weights (average=False) --> [B, num_queries, source_size]
            # memory_attention_weights (average=True) --> [B, num_heads, num_queries, source_size]
            queries_attn_maps = [weights[0][i].reshape(16, 16) for i in range(num_queries)]
            self.attn_maps.append(queries_attn_maps)
        # normalize and return
        output = self.norm(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # multihead self-attention layers
        # batch first --> [B, num_queries, C]
        self.queries_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True
        )
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True
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

    def forward(self, input, memory, pos: Optional[Tensor], queries_pos: Optional[Tensor]):
        # # INPUT SHAPE: [B, num_queries, C]
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
        k_memory = self.with_pos_embed(memory, pos)
        q_memory = self.with_pos_embed(add_norm_out, queries_pos)
        # compute self-attention
        memory_attention_out, memory_attention_weights = self.memory_attention(q_memory, k_memory, v_memory, need_weights=True, average_attn_weights=True)
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
        return result, memory_attention_weights