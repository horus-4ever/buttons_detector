import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiscaleResNet50(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.body = create_feature_extractor(
            resnet,
            return_nodes={
                "layer1": "feat_s4",
                "layer2": "feat_s8",
                "layer4": "feat_s32",
            },
        )

        self.proj_s4 = ConvNormAct(256, hidden_dim)
        self.proj_s8 = ConvNormAct(512, hidden_dim)
        self.proj_s32 = ConvNormAct(2048, hidden_dim)

    def forward(self, x):
        """
        Extract layers 1, 2 and 4 from the ResNet50 backbone.
        - x: [B, C, W, H]

        - output: (features l * [B, embed_dim, Hl, Wl], spatial_shapes l * [2])
        """
        B, C, W, H = x.size()
        features = self.body(x)

        feat_s4 = features["feat_s4"]      # [B, 256,  H/4,  W/4]
        feat_s8 = features["feat_s8"]      # [B, 512,  H/8,  W/8]
        feat_s32 = features["feat_s32"]    # [B, 2048, H/32, W/32]

        feat_s4 = self.proj_s4(feat_s4)    # [B, hidden_dim, H/4, W/4]
        feat_s8 = self.proj_s8(feat_s8)    # [B, hidden_dim, H/8, W/8]
        feat_s32 = self.proj_s32(feat_s32) # [B, hidden_dim, H/32, W/32]

        features = [
            feat_s4, feat_s8, feat_s32
        ]

        return features
    

if __name__ == "__main__":
    model = MultiscaleResNet50()
    input = torch.rand(3 * 256 * 256).view(1, 3, 256, 256)
    f4, f8, f32 = model(input)
    print(f4.size(), f8.size(), f32.size())