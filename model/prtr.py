import torch
from torch import nn
from .transformer import DeformableTransformer
import torch.nn.functional as F
from .position_encoding import PositionEmbeddingSine2D
from .backbone import MultiscaleResNet50
from .config import ModelConfig
from utils import MLP, inverse_sigmoid
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
            nrefpointsperquery: int = 2,
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
        for p in self.backbone.body.parameters():
            p.requires_grad = False
            
        # construct the transformer
        self.transformer = DeformableTransformer(
            d_model=self.d_model,
            nheads=n_heads,
            nlevels=4,
            npoints=n_points,
            nrefpointsperquery=nrefpointsperquery,
            encoder_nlayers=n_encoder_layers,
            decoder_nlayers=n_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        self.query_embed = nn.Embedding(num_queries, self.d_model)
        # NEW: The reference point embedding is there to encode the difference between a button and a hole.
        #      It is applied to all reference points on top of the query_embed, which encodes the pair information.
        self.refpoints_embed = nn.Embedding(nrefpointsperquery, self.d_model)
        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=self.d_model // 2)
        self.class_head = nn.Linear(self.d_model, num_classes + 1)
        # NEW: the button head now predicts the bounding box
        self.button_head = MLP(self.d_model, mlp_hidden_dim, 4, mlp_num_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.body.eval()
        return self
        
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

        masks = masks.to(device=inputs.device)

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

        hs, encoder_attn_maps, decoder_attn_maps, spatial_shapes, encoder_sampling_locations, decoder_sampling_locations, reference_points = self.transformer(
            features=feature_maps, # num_levels * [B, embed_dim, Hl, Wl]
            masks=multilevel_masks, # num_levels * [B, 1, Hl, Wl]
            pos_embeds=position_embeddings, # num_levels * [B, embed_dim, Hl, Wl]
            query_embed=self.query_embed.weight, # [num_queries, embed_dim]
            refpoints_embed=self.refpoints_embed.weight, # [RpQ, embed_dim]
        )
        # hs: [B, query_len, embed_dim]
        # reference_points: [num_queries, RpQ, 4]
        Q, num_ref_points_per_query, _ = reference_points.size()
        # NEW: the button head now predicts the bounding box
        # -> [B, Q, embed_dim]
        class_head_input = hs.mean(dim=2) # take the mean over RpQ
        pred_logits = self.class_head(class_head_input) # [B, num_queries, num_classes+1]
        # for the buttons we of course need to keep by RpQ
        # [B, query_len, RpQ, 4]
        button_deltas = self.button_head(hs) # [B, num_queries, 4]
        # take the button centers, which are the first two coordinates
        button_centers = button_deltas[..., :2] # [B, num_queries, 2]
        pred_buttons_centers = (inverse_sigmoid(reference_points) + button_centers).sigmoid()  # [B, num_queries, 2]
        # now the width and height is given by the last two coordinates
        pred_buttons_wh = button_deltas[..., 2:].sigmoid()  # [B, num_queries, 2]
        pred_boxes = torch.cat([pred_buttons_centers, pred_buttons_wh], dim=-1)  # [B, num_queries, 4]

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "encoder_attn_maps": encoder_attn_maps, # encoder_layers * [batch, sum_l(Hl * Wl), heads, num_levels, num_points]
            "encoder_sampling_locations": encoder_sampling_locations, # [batch, sum_l(Hl * Wl), heads, num_levels, num_points, 2]
            "decoder_attn_maps": decoder_attn_maps, # decoder_layers * [batch, num_queries * RpQ, heads, num_levels, num_points]
            "decoder_sampling_locations": decoder_sampling_locations, # [batch, num_queries, heads, num_levels, num_points, 2]
            "spatial_shapes": spatial_shapes, # [num_levels, 2]
            "reference_points": reference_points, # [query_len, RpQ, 2]
            "image_size": (H, W)
        }




def build_model(model_config: ModelConfig):
    model_params = model_config.model_parameters
    model = PRTR(
        name=model_config.name,
        num_classes=1,
        num_queries=model_params.num_queries,
        d_model=model_params.d_model,
        n_heads=model_params.n_heads,
        n_encoder_layers=model_params.n_encoder_layers,
        n_decoder_layers=model_params.n_decoder_layers,
        n_points=model_params.n_points,
        nrefpointsperquery=2,
        dim_ffn=model_params.dim_ffn,
        dropout=model_params.dropout,
        activation=model_params.activation,
        mlp_hidden_dim=model_params.mlp_hidden_dim,
        mlp_num_layers=model_params.mlp_num_layers
    )
    return model


def build_model_from(json_path: str):
    path = Path(json_path)
    model_config = ModelConfig.open(path)
    return build_model(model_config)
