from torch import nn, Tensor
import torch
import math


class PositionEmbeddingSine2D(nn.Module):
    def __init__(
            self,
            num_pos_feats: int,
            temperature: float = 10_000.0,
            normalize: bool = True,
            scale: float = 2 * math.pi
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale # scale is used for normalization

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        # create the 2D coordinates grid
        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)  # [H, W]
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)  # [H, W]
        # optional normalization part
        if self.normalize:
            eps = 1e-6
            if H > 1:
                y_embed = y_embed / (H - 1 + eps) * self.scale
            if W > 1:
                x_embed = x_embed / (W - 1 + eps) * self.scale
        # create the i
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_y = y_embed[:, :, None] / dim_t
        pos_x = x_embed[:, :, None] / dim_t

        # Apply sin to even indices, cos to odd indices
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        # Concatenate y and x encodings: [H, W, 2*num_pos_feats]
        pos = torch.cat((pos_y, pos_x), dim=2)

        # Rearrange to [B, 2*num_pos_feats, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

        return pos
    

if __name__ == "__main__":
    proj = torch.randn(2, 256, 20, 30)   # [B, C, H, W]
    pos_embed = PositionEmbeddingSine2D(num_pos_feats=128)
    pos = pos_embed(proj)

    print(pos.shape)   # [2, 256, 20, 30]
