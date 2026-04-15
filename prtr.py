import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from transformer import Transformer
from position_encoding import PositionEmbeddingSine2D
from utils import MLP
from pathlib import Path
import json


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
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
        # freeze the backbone
        for p in self.backbone.parameters():
            p.require_grad = False # type: ignore
        # construct the transformer
        self.conv = nn.Conv2d(2048, self.d_model, 1)
        self.transformer = Transformer(
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
        

    def forward(self, inputs):
        # inputs: [B, 3, H_img, W_img]
        B = inputs.shape[0]

        # 1) Backbone
        x = self.backbone(inputs)          # [B, 2048, H, W]

        # 2) Project to transformer dim
        x = self.conv(x)                   # [B, 256, H, W]

        # 3) Positional encoding on feature map
        pos = self.position_embedding(x)   # [B, 256, H, W]

        # pos = pos.flatten(2).transpose(1, 2)   # [B, H*W, 256]

        # 7) Transformer
        hs, memory = self.transformer(
            src=x,
            pos_embed=pos,
            query_embed=self.query_embed.weight
        )  # expected [B, num_queries, 256] or similar depending on your Transformer
        
        pred_logits = self.class_head(hs)         # [B, num_queries, num_classes+1]
        pred_buttons = self.button_head(hs).sigmoid()  # [B, num_queries, 2]

        return {
            "pred_logits": pred_logits,
            "pred_buttons": pred_buttons,
            "memory": memory
        }




class PRTRSingleHead(nn.Module):
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
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
        # freeze the backbone
        for p in self.backbone.parameters():
            p.require_grad = False # type: ignore
        # construct the transformer
        self.conv = nn.Conv2d(2048, self.d_model, 1)
        self.transformer = Transformer(
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
        # we predict the hole in the same prediction head
        self.button_head = MLP(self.d_model, mlp_hidden_dim, 4, mlp_num_layers)
        

    def forward(self, inputs):
        # inputs: [B, 3, H_img, W_img]
        B = inputs.shape[0]

        # 1) Backbone
        x = self.backbone(inputs)          # [B, 2048, H, W]

        # 2) Project to transformer dim
        x = self.conv(x)                   # [B, 256, H, W]

        # 3) Positional encoding on feature map
        pos = self.position_embedding(x)   # [B, 256, H, W]
        
        # 7) Transformer
        hs, memory = self.transformer(
            src=x,
            pos_embed=pos,
            query_embed=self.query_embed.weight
        )  # expected [B, num_queries, 256] or similar depending on your Transformer
        
        pred_logits = self.class_head(hs)         # [B, num_queries, num_classes+1]
        pred_buttons = self.button_head(hs).sigmoid()  # [B, num_queries, 2]

        return {
            "pred_logits": pred_logits,
            "pred_buttons": pred_buttons,
            "memory": memory
        }




def build_model_from(json_path: str):
    path = Path(json_path)
    with open(path, "r") as file:
        data = json.load(file)
        model_name = data["model_name"]
        parameters = data["parameters"]
        return PRTR(model_name, **parameters)


if __name__ == "__main__":    
    pass