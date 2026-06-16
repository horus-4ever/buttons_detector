import torch
from torch import nn
from transformer import DeformableTransformer
import torch.nn.functional as F
from position_encoding import PositionEmbeddingSine2D
from backbone import MultiscaleResNet50
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
            n_points: int = 4,
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
        self.backbone = MultiscaleResNet50(hidden_dim=self.d_model)

        # freeze the backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 
        # construct the transformer
        self.transformer = DeformableTransformer(
            d_model=self.d_model,
            nheads=n_heads,
            nlevels=3,
            npoints=n_points,
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
        
    def _compute_masks(self, masks, feature_maps):
        """
        - masks: [B, H_img, W_img]
        - feature_maps: l * [B, embed_dim, Hl, Wl]

        - output: l * [B, 1, Hl, Wl]
        """
        B, H_img, W_img = masks.size()
        output = []
        for feature_map in feature_maps:
            # feature_map: [B, embed_dim, Hl, Wl]
            _, _, H, W = feature_map.size()
            height_indices = torch.tensor(torch.round(torch.linspace(0.5 / H, (1.0 - 0.5 / H), H) * H_img), dtype=torch.int, device=feature_map.device)
            width_indices = torch.tensor(torch.round(torch.linspace(0.5 / W, 1.0 - 0.5 / W, W) * W_img), dtype=torch.int, device=feature_map.device)
            h_indices, w_indices = torch.meshgrid(height_indices, width_indices, indexing='ij')
            # [B, Hl, Wl]
            level_map = masks[:, h_indices, w_indices]
            # [B, Hl, Wl] -> [B, 1, Hl, Wl]
            level_map = level_map[:, None, :, :]
            output.append(level_map)
        return output


    def forward(self, inputs, masks):
        """
        - inputs: [B, 3, H_img, W_img]
        - masks: [B, H_img, W_img] (binary mask containing 1 on padded areas and 0 on non-padded areas)

        Inputs are padded to the maximum batch size.
        Regarding the masks, we get in input the original image mask.
        Masks for the extracted multiscale features maps will be derived from the original image mask.
        """
        # inputs: [B, 3, H_img, W_img]
        B, _, H, W = inputs.shape

        # feature_maps: l * [B, embed_dim, Hl, Wl]
        feature_maps = self.backbone(inputs)
        # masks: l * [B, 1, Hl, Wl]
        multilevel_masks = self._compute_masks(masks, feature_maps)

        position_embeddings = []
        for mask in multilevel_masks:
            mask = mask.permute(0, 2, 3, 1).contiguous().squeeze(-1)
            # [B, embed_dim, Hl, Wl]
            pos_embed = self.position_embedding(mask)
            position_embeddings.append(pos_embed)

        hs, memory, attn_maps = self.transformer(
            features=feature_maps, # num_levels * [B, embed_dim, Hl, Wl]
            masks=multilevel_masks, # num_levels * [B, 1, Hl, Wl]
            pos_embeds=position_embeddings, # num_levels * [B, embed_dim, Hl, Wl]
            query_embed=self.query_embed.weight, # [num_queries, embed_dim]
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