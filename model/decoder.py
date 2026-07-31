from torch import nn, Tensor
import torch.nn.functional as F
from model.utils import FFN
import torch
from typing import Optional
from .attention import MultiscaleDeformableAttention, MultiscaleMultireferencesDeformableAttention


class Decoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            nlayers: int,
            nlevels: int,
            npoints: int,
            nrefpointsperquery: int,
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
                nrefpointsperquery=nrefpointsperquery,
                dim_ffn=dim_ffn,
                dropout=dropout,
                activation=activation
            )
            for _ in range(nlayers)
        ])
        self.num_ref_points_per_query = nrefpointsperquery
        # normalization layer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, memory, reference_points, spatial_shapes, query_embed: Optional[Tensor], refpoints_embed, memory_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [B, num_queries, embed_dim]
        - memory: [B, sum_l(Hl * Wl), embed_dim]
        - reference_points: [num_queries, 2]
        - spatial_shapes: [num_levels, 2]
        - pos: [B, suml(Hl * Wl), embed_dim]
        - query_embed: [num_queries, embed_dim]
        - refpoints_embed: [RpQ, embed_dim]
        - memory_key_padding_mask: [B, 1, suml(Hl * Wl)]
        """
        decoder_attn_weights = []
        decoder_sampling_locations = []
        # loop over the decoder layers
        B, Q, C = input.size()
        output = input[:, :, None, :].expand(B, Q, self.num_ref_points_per_query, C).contiguous()
        for layer in self.layers:
            output, attn_weights, sampling_locations = layer(
                input=output,
                memory=memory,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                query_embed=query_embed,
                refpoints_embed=refpoints_embed, # [RpQ, embed_dim]
                memory_key_padding_mask=memory_key_padding_mask # [B, 1, sum_l(Hl, Wl)]
            )
            decoder_attn_weights.append(attn_weights)
            decoder_sampling_locations.append(sampling_locations)
        # decoder_attn_weights: decoder_layers * [batch, query_len, heads, num_levels, num_points]
        # normalize and return
        output = self.norm(output)
        return output, decoder_attn_weights, decoder_sampling_locations


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            nlevels: int,
            npoints: int,
            nrefpointsperquery: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        self.num_levels = nlevels
        self.num_ref_points_per_query = nrefpointsperquery
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
            # num_ref_points_per_query=nrefpointsperquery # NEW: number of reference points per query
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
    
    def with_queries_embed(self, tensor, query_embed: Tensor, refpoints_embed: Tensor):
        """
        - tensor: [..., num_queries * RqP, embed_dim]
        - query_embed: [num_queries, embed_dim]
        - refpoints_embed: [RqP, embed_dim]

        - output: [B, num_queries * RqP, embed_dim]
        """
        Q, C = query_embed.size()
        # first reshape the query embed correctly
        # [num_queries, embed_dim] -> [num_queries, RqP, embed_dim]
        query_embed = query_embed.unsqueeze(1).expand(Q, self.num_ref_points_per_query, C).contiguous()
        # [num_queries, RqP, embed_dim] -> [num_queries * RqP, embed_dim]
        query_embed = query_embed.view(Q * self.num_ref_points_per_query, C)
        # then reshape the refpoints embed correctly
        # [RqP, embed_dim] -> [num_queries, RqP, embed_dim]
        refpoints_embed = refpoints_embed.unsqueeze(0).expand(Q, self.num_ref_points_per_query, C).contiguous()
        # [num_queries, RqP, embed_dim] -> [num_queries * RqP, embed_dim]
        refpoints_embed = refpoints_embed.view(Q * self.num_ref_points_per_query, C)
        return tensor + query_embed + refpoints_embed

    def forward(self, input, memory, reference_points, spatial_shapes, query_embed: Tensor, refpoints_embed, memory_key_padding_mask: Optional[Tensor] = None):
        """
        - input: [B, num_queries, RpQ, embed_dim]
        - memory: [B, query_len, embed_dim]
        - reference_points: [query_len, RpQ, 2]
        - query_embed: [num_queries, embed_dim]
        - refpoints_embed: [RpQ, embed_dim]
        - memory_key_padding_mask: [B, 1, sum_l(Hl, Wl)]
        """
        B, Q, RQP, C = input.size()
        # [B, num_queries, RpQ, embed_dim] -> [B, RpQ * num_queries, embed_dim]
        input = input.view(B, Q * RQP, C)
        # computes k and q for queries attention
        # [B, RpQ * num_queries, embed_dim]
        k_queries = q_queries = self.with_queries_embed(input, query_embed, refpoints_embed)
        v_queries = input
        # compute self-attention on queries and dropout
        queries_attention_out = self.queries_attention(
            q_queries, k_queries, v_queries,
            key_padding_mask=None
        )[0]
        queries_attention_out = self.dropout1(queries_attention_out)
        # add and normalize
        # add_norm_out: [B, num_queries * RpQ, embed_dim]
        add_norm_out = input + queries_attention_out
        add_norm_out = self.norm1(add_norm_out)

        # now, the decoder part
        # computes v, k and q for memory attention
        v_memory = memory # [B, query_len, embed_dim]
        # [B, num_queries * RqP, embed_dim]
        q_memory = self.with_queries_embed(add_norm_out, query_embed, refpoints_embed)
        # resize the reference_points to the right size
        # [query_len, RpQ, 2] -> [query_len * RpQ, 2]
        reference_points = reference_points.view(Q * self.num_ref_points_per_query, 2)
        # [query_len * RpQ, 2] -> [batch, query_len * RpQ, num_levels, 2]
        reference_points = reference_points[None, :, None, :].expand(B, Q * self.num_ref_points_per_query, self.num_levels, 2)
        # compute self-attention
        memory_attention_out, memory_attention_weights, memory_attention_sampling_locations = self.memory_attention(
            query=q_memory, # [B, num_queries, embed_dim]
            reference_points=reference_points, # [batch, query_len * RpQ, num_levels, 2]
            values=v_memory, # [B, query_len, embed_dim]
            spatial_shapes=spatial_shapes, # [num_levels, 2]
            key_padding_mask=memory_key_padding_mask, # [B, 1, sum_l(Hl, Wl)]
        )
        # memory_attention_weights: [batch, query_len, heads, num_levels, num_points]
        # result: [B, Q * RpQ, embed_dim]
        memory_attention_out = self.dropout2(memory_attention_out)
        # add and normalize
        # add_norm_out: [B, num_queries * RpQ, embed_dim]
        # memory_attention_out: [B, Q * RpQ, embed_dim]
        # [B, num_queries * RpQ, embed_dim]
        add_norm_out = add_norm_out + memory_attention_out
        add_norm_out = self.norm2(add_norm_out)
        # ffn
        ffn_out = self.ffn(add_norm_out)
        ffn_out = self.dropout3(ffn_out)
        # add and normalize
        result = add_norm_out + ffn_out
        result = self.norm3(result) # [B, Q * RpQ, embed_dim]
        result = result.view(B, Q, self.num_ref_points_per_query, C)
        return result, memory_attention_weights, memory_attention_sampling_locations