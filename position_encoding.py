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

    def forward(self, mask: Tensor) -> Tensor:
        not_mask = ~mask   # valid pixels
        dtype = torch.float32
        device = mask.device

        # cumulative coordinates over valid region only
        y_embed = not_mask.cumsum(1, dtype=dtype)
        x_embed = not_mask.cumsum(2, dtype=dtype)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=dtype, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_y = y_embed[:, :, :, None] / dim_t   # [B, H, W, F]
        pos_x = x_embed[:, :, :, None] / dim_t   # [B, H, W, F]

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)   # [B, H, W, 2F]
        pos = pos.permute(0, 3, 1, 2)            # [B, 2F, H, W]

        return pos
    

if __name__ == "__main__":
    proj = torch.randn(2, 256, 20, 30)   # [B, C, H, W]
    pos_embed = PositionEmbeddingSine2D(num_pos_feats=128)
    pos = pos_embed(proj)

    print(pos.shape)   # [2, 256, 20, 30]
