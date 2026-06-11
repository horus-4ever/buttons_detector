import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from transformer import DeformableTransformer
import torch.nn.functional as F
from position_encoding import PositionEmbeddingSine2D
from utils import MLP
from pathlib import Path
import json


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
        features = self.body(x)

        feat_s4 = features["feat_s4"]      # [B, 256,  H/4,  W/4]
        feat_s8 = features["feat_s8"]      # [B, 512,  H/8,  W/8]
        feat_s32 = features["feat_s32"]    # [B, 2048, H/32, W/32]

        feat_s4 = self.proj_s4(feat_s4)    # [B, hidden_dim, H/4, W/4]
        feat_s8 = self.proj_s8(feat_s8)    # [B, hidden_dim, H/8, W/8]
        feat_s32 = self.proj_s32(feat_s32) # [B, hidden_dim, H/32, W/32]

        target_size = feat_s4.shape[-2:] # (H/4, W/4)

        feat_s8 = F.interpolate(
            feat_s8,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        feat_s32 = F.interpolate(
            feat_s32,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        # return the feature maps
        return [feat_s4, feat_s8, feat_s32]


class PRTR(nn.Module):
    def __init__(
            self,
            name: str,
            num_classes: int = 2,
            num_queries: int = 10,
            d_model: int = 128,
            n_heads: int = 4,
            n_encoder_layers: int = 3,
            n_decoder_layers: int = 1,
            dim_ffn: int = 512,
            dropout: float = 0.1,
            activation: str = "relu",
            mlp_hidden_dim: int = 128,
            mlp_num_layers: int = 3
        ):
        """Regarding the number of classes, the no-class is added so it must not be counted in."""
        super().__init__()
        self.name = name
        self.d_model = d_model
        # take backbone from resnet-50 model
        self.backbone = MultiscaleResNet50(hidden_dim=256)

        # freeze the backbone
        for p in self.backbone.parameters():
            p.require_grad = False # type: ignore
        # construct the transformer
        self.transformer = DeformableTransformer(
            d_model=self.d_model,
            nheads=n_heads,
            encoder_nlayers=n_encoder_layers,
            decoder_nlayers=n_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        self.query_embed = nn.Embedding(num_queries, self.d_model)
        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=self.d_model // 2)
        self.class_head = nn.Linear(self.d_model, num_classes + 1)
        self.button_head = MLP(self.d_model, mlp_hidden_dim, 2, mlp_num_layers)
        

    def forward(self, inputs, masks):
        """
        - inputs: [B, 3, H_img, W_img]
        - masks: [B, 1, H_img, W_img] (binary mask containing 1 on padded areas and 0 on non-padded areas)
        """
        # inputs: [B, 3, H_img, W_img]
        B, _, H, W = inputs.shape

        # let's call H~ = H/4 and W~ = W/4 for simplicity
        feature_maps, positions = self.backbone(inputs)          # [B, 256, H~, W~]

        hs, memory, attn_maps = self.transformer(
            src=feature_maps,
            masks=masks,
            pos_embed=positions,
            query_embed=self.query_embed.weight
        )
        
        pred_logits = self.class_head(hs)         # [B, num_queries, num_classes+1]
        pred_buttons = self.button_head(hs).sigmoid()  # [B, num_queries, 2]

        return {
            "pred_logits": pred_logits,
            "pred_buttons": pred_buttons,
            "memory": memory,
            "attn_maps": attn_maps,
            "image_size": (H, W)
        }




def build_model_from(json_path: str):
    path = Path(json_path)
    with open(path, "r") as file:
        data = json.load(file)
        model_name = data["model_name"]
        parameters = data["parameters"]
        return PRTR(model_name, **parameters)


if __name__ == "__main__":    
    model = PRTR("test_model")
    dummy_input = torch.randn(2, 3, 256, 192)
    outputs = model(dummy_input)
    print(outputs["pred_logits"].shape)  # Expected: [B, num_queries, num_classes+1]
    print(outputs["pred_buttons"].shape)  # Expected: [B, num_queries, 2]
    print(outputs["memory"].shape)        # Expected: [B, C, H, W]
    print(len(outputs["attn_maps"]))      # Expected: number of attention maps returned by the transformer