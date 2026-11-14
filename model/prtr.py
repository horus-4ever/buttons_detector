import torch
from torch import nn
from .transformer import DeformableTransformer
import torch.nn.functional as F
from .position_encoding import PositionEmbeddingSine2D
from .backbone import MultiscaleResNet50
from .config import ModelConfig
from model.utils import MLP, inverse_sigmoid
from pathlib import Path
from .mode import Mode


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
        for p in self.backbone.body.parameters():
            p.requires_grad = False
            
        # construct the transformer
        self.transformer = DeformableTransformer(
            d_model=self.d_model,
            nheads=n_heads,
            nlevels=4,
            npoints=n_points,
            encoder_nlayers=n_encoder_layers,
            decoder_nlayers=n_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        self.query_embed = nn.Embedding(num_queries, self.d_model * 2)
        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=self.d_model // 2)
        self.class_head = nn.Linear(self.d_model, num_classes + 1)
        # NEW: the button head now predicts the bounding box
        self.pred_head = MLP(self.d_model, mlp_hidden_dim, 8, mlp_num_layers)
        self._reset_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.body.eval()
        return self
    
    def _reset_parameters(self):
        # initialization of the prediction head
        # targets are very small so we want to output very small boxes at first
        # as for the initial x and y, the predictions are offsets from reference points
        # therefore, we want x and y offsets to be null at first, or at least a small value
        last_layer: nn.Linear = self.pred_head.layers[-1] # type: ignore
        with torch.no_grad():
            last_layer.weight.zero_()
            last_layer.bias.zero_()
            initial_value = last_layer.bias.new_tensor([0.02, 0.02]) # ~ 10 pixels
            # the width and height are taken with a .sigmoid, so we perform a .logit here
            last_layer.bias[2:4].copy_(torch.logit(initial_value))
            last_layer.bias[6:8].copy_(torch.logit(initial_value))
        
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
            size = (H, W)
            level_mask = F.interpolate(
                masks[:, None].float(),
                size=size,
                mode="nearest",
            ).to(torch.bool)
            output.append(level_mask)
        return output


    def forward(self, inputs, masks, mode: Mode = Mode.TRAINING):
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
            query_embed=self.query_embed.weight, # [num_queries, embed_dim * 2]
        )
        # hs: [B, query_len, embed_dim]
        # reference_points: [num_queries, 2]
        Q, _ = reference_points.size()
        # -> [B, Q, embed_dim]
        pred_logits = self.class_head(hs) # [B, num_queries, num_classes+1]
        # [B, query_len 8]
        deltas = self.pred_head(hs) # [B, Q, 8]
        # take the button centers, which are the first two coordinates
        button_centers = deltas[..., :2] # [B, Q, 2]
        counterpart_centers = deltas[..., 4:6] # [B, Q, 2]
        pred_button_centers = (inverse_sigmoid(reference_points) + button_centers).sigmoid()  # [B, Q, 2]
        pred_counterpart_centers = (inverse_sigmoid(reference_points) + counterpart_centers).sigmoid() # [B, Q, 2]
        # now the width and height is given by the last two coordinates
        pred_button_whs = deltas[..., 2:4].sigmoid()  # [B, Q, 2]
        pred_counterpart_whs = deltas[..., 6:].sigmoid()  # [B, Q, 2]
        pred_boxes = torch.cat([pred_button_centers, pred_button_whs, pred_counterpart_centers, pred_counterpart_whs], dim=-1)  # [B, Q, 8]
        # [B, Q, 8] -> [B, Q, 2, 4]
        pred_boxes = pred_boxes.view(B, Q, 2, 4)

        if mode == Mode.DEBUG:
            return {
                "pred_logits": pred_logits, # [B, num_queries, num_classes+1]
                "pred_boxes": pred_boxes, # [B, Q, 2, 4]
                "encoder_attn_maps": encoder_attn_maps, # encoder_layers * [batch, sum_l(Hl * Wl), heads, num_levels, num_points]
                "encoder_sampling_locations": encoder_sampling_locations, # [batch, sum_l(Hl * Wl), heads, num_levels, num_points, 2]
                "decoder_attn_maps": decoder_attn_maps, # decoder_layers * [batch, num_queries, heads, num_levels, num_points]
                "decoder_sampling_locations": decoder_sampling_locations, # [batch, num_queries, heads, num_levels, num_points, 2]
                "spatial_shapes": spatial_shapes, # [num_levels, 2]
                "reference_points": reference_points, # [query_len, 2]
                "image_size": (H, W)
            }
        elif mode == Mode.EVALUATION:
            return {
                "pred_logits": pred_logits, # [B, num_queries, num_classes+1]
                "pred_boxes": pred_boxes, # [B, Q, 2, 4],
                "image_size": (H, W)
            }
        elif mode == Mode.TRAINING:
            return {
                "pred_logits": pred_logits, # [B, num_queries, num_classes+1]
                "pred_boxes": pred_boxes, # [B, Q, 2, 4],
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
        dim_ffn=model_params.dim_ffn,
        dropout=model_params.dropout,
        activation=model_params.activation,
        mlp_hidden_dim=model_params.mlp_hidden_dim,
        mlp_num_layers=model_params.mlp_num_layers
    )
    return model


def build_model_from(json_path: str | Path):
    path = Path(json_path)
    model_config = ModelConfig.open(path)
    return build_model(model_config)


def load_model_from_weights(weights: Path, device):
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    assert "model_state_dict" in checkpoint
    model_config = ModelConfig.from_json(checkpoint["model_configuration"])
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model