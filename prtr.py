import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from transformer import Transformer
from position_encoding import PositionEmbeddingSine2D
from utils import MLP


class PRTR(nn.Module):
    def __init__(self, num_classes: int, num_queries: int):
        super().__init__()
        self.d_model = 128
        # take backbone from resnet-50 model
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
        # freeze the backbone
        for p in self.backbone.parameters():
            p.require_grad = False
        self.conv = nn.Conv2d(2048, self.d_model, 1)
        self.transformer = Transformer(
            d_model=self.d_model,
            nheads=4,
            encoder_nlayers=3,
            decoder_nlayers=1,
            dim_ffn=512,
            dropout=0.1,
            activation="relu"
        )
        self.query_embed = nn.Embedding(num_queries, self.d_model)
        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=self.d_model // 2)
        self.class_head = nn.Linear(self.d_model, num_classes + 1)
        self.button_head = MLP(self.d_model, self.d_model, 2, num_layers=3)
        

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


if __name__ == "__main__":    
    detr = PRTR(num_classes=5, num_queries=10)
    detr.eval()
    inputs = torch.randn(1, 3, 1024, 1024)

    outputs = detr(inputs)
    print(outputs)

    """
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50')
    model.eval()

    outputs = model(inputs)
    print(outputs["pred_logits"])
    print("=========== == == == == == == ==========")
    print(outputs["pred_boxes"])
    """
