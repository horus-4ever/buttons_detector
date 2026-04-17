from torch import nn, Tensor
import torch.nn.functional as F
import torch
from utils import FFN
from typing import Optional


class Encoder(nn.Module):
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
        # module of nlayers encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
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

    def forward(self, input, pos: Optional[Tensor], src_key_padding_mask: Optional[Tensor] = None):
        result = input
        for layer in self.encoder_layers:
            result = layer(result, pos=pos, src_key_padding_mask=src_key_padding_mask)
        # normalize
        result = self.norm(result)
        return result


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nheads: int,
            dim_ffn: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # multihead self-attention module
        self.multi_head_self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True
        )
        # feed-forward network
        self.ffn = FFN(d_model, dim_ffn, dropout, activation=activation)
        # normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, input, pos: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None):
        # compute Q and K matrices and apply positional embedding to it
        q = k = self.with_pos_embed(input, pos)
        # compute self-attention and dropout
        self_att_out = self.multi_head_self_attention(q, k, value=input, key_padding_mask=src_key_padding_mask)[0]
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
        return result

        

if __name__ == "__main__":
    # test of the layers
    B = 2        # batch size
    S = (1024 // 32) ** 2      # number of image tokens = H*W
    C = 256      # token dimension

    src = torch.randn(B, S, C)   # image features after backbone + 1x1 projection + flatten
    pos = torch.randn(B, S, C)   # 2D positional encoding

    encoder = Encoder(
        d_model=256,
        nheads=8,
        nlayers=6,
        dim_ffn=2048,
        dropout=0.1,
        activation="relu",
    )

    memory = encoder(src, pos=pos)   # [B, S, C]
    print(memory.shape)