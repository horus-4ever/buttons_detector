from dataclasses import dataclass
from typing import Any
from PIL import ImageDraw
import numpy as np
from dataformat.dataformat import BoundingBox
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from model.mode import Mode


@dataclass
class Prediction:
    class_id: int
    button_bbox: BoundingBox
    counterpart_bbox: BoundingBox
    confidence: float
    query: int


@dataclass
class AttentionImage:
    image: Image.Image
    query: int


@dataclass
class InferenceResult:
    predictions: list[Prediction]
    prediction_image: Image.Image
    attention_images: Any


class Inference:
    PAIR_CLASS_ID = 0

    def __init__(self, model, device, image_size = 512):
        self.model = model
        self.device = device
        self.image_size = image_size

    @torch.no_grad()
    def run_one(self, image: Image.Image, attention: bool = False):
        self._set_eval_mode()
        width, height = image.size
        processed_image = self._preprocess_image(image, self.image_size).to(device=self.device)
        masks = torch.zeros([1, height, width], dtype=torch.bool)
        inference_mode = Mode.DEBUG
        with torch.inference_mode(True):
            results = self.model(processed_image, masks, mode=inference_mode)
            inference_result = self._get_results(image, results)
            return inference_result

    def _get_results(self, image: Image.Image, results: dict):
        predictions = self._get_predictions(results)
        attention_image = self._visualize_decoder_attention(predictions, results)
        pred_attention_image = self._visualize_pred_decoder_attention(predictions, results)
        predictions_image = self._visualize_predictions(image, predictions)
        refpoints_image = self._visualize_reference_points(results)
        self._sampling_points_distance_pred(predictions, results)
        return {
            "predictions": predictions,
            "decoder_attention_image": attention_image,
            "pred_decoder_attention_image": pred_attention_image,
            "predictions_image": predictions_image,
            "refpoints_image": refpoints_image
        }

    def _preprocess_image(self, image: Image.Image, inference_size: int) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((inference_size, inference_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return transform(image).unsqueeze(0) # type: ignore

    def _set_eval_mode(self):
        self.model.eval()

    def normalized_to_pixel_xy(self, xy: torch.Tensor, width: int, height: int) -> tuple[float, float, float, float]:
        x = float(xy[0]) * width
        y = float(xy[1]) * height
        w = float(xy[2]) * width
        h = float(xy[3]) * height
        return x, y, w, h

    def _get_predictions(self, outputs):
        height, width = outputs["image_size"]
        # put on the CPU, and we have only one batch
        pred_logits = outputs["pred_logits"][0].detach().cpu() # [Q, C + 1]
        pred_boxes = outputs["pred_boxes"][0].detach().cpu() # [Q, 2, 4]
        pred_buttons = pred_boxes[:, 0, :] # [Q, 4]
        pred_counterparts = pred_boxes[:, 1, :] # [Q, 4]
        # transforms to probabilities
        pred_probs = pred_logits.softmax(dim=-1) # [Q, C + 1]
        pred_classes = pred_probs.argmax(dim=-1)
        pred_scores = pred_probs.max(dim=-1).values
        # now loop over the queries and get the predictions
        num_queries, num_classes = pred_logits.size()
        predictions = []
        for query_idx in range(num_queries):
            class_id = int(pred_classes[query_idx])
            confidence = float(pred_probs[query_idx, self.PAIR_CLASS_ID])
            b_xy_norm_tensor = pred_buttons[query_idx]
            f_xy_norm_tensor = pred_counterparts[query_idx]
            b_pos_x, b_pos_y, b_bbox_w, b_bbox_h = self.normalized_to_pixel_xy(b_xy_norm_tensor, width, height)
            f_pos_x, f_pos_y, f_bbox_w, f_bbox_h = self.normalized_to_pixel_xy(f_xy_norm_tensor, width, height)
            button_bbox = BoundingBox(b_pos_x, b_pos_y, b_bbox_w, b_bbox_h)
            fastener_bbox = BoundingBox(f_pos_x, f_pos_y, f_bbox_w, f_bbox_h)
            predictions.append(Prediction(class_id, button_bbox, fastener_bbox, confidence, query_idx))
        return predictions

    def _visualize_reference_points(self, results: dict):
        """
        Visualize the reference points.
        """
        # [Q, RpQ, 2]
        reference_points = results["reference_points"]
        color = (0, 255, 0, 255)
        radius = 5
        # now create the image
        result_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        blackboard = ImageDraw.Draw(result_image)
        for points in reference_points:
            posx, posy = points
            posx, posy = posx * 512, posy * 512
            blackboard.ellipse(
                (posx - radius, posy - radius, posx + radius, posy + radius),
                fill=color
            )
        return result_image

    def _visualize_decoder_attention(self, predictions: list[Prediction], results: dict, layer: int = -1, separated = True):
        """
        Visualize the decoder attention.
        """
        # get the attention maps and the sampling locations
        attn_maps = results["decoder_attn_maps"][layer][0] # last decoder layer ; first image of batch
        # n * [B, Q, n_heads, n_levels, n_points, 2]
        sampling_locations = results["decoder_sampling_locations"][layer][0] # [Q, n_heads, n_levels, n_points, 2]
        Q, n_heads, n_levels, n_points, _ = sampling_locations.size()
        the_constant = n_heads * n_levels * n_points
        sampling_locations = sampling_locations.reshape(-1, 2) # [Q * n_heads * n_levels * n_points, 2]
        attn_maps = attn_maps.reshape(-1, 1) # [Q * n_heads * n_levels * num_points, 1]
        # create the images
        result_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        for i, (attn, (posx, posy)) in enumerate(zip(attn_maps, sampling_locations)):
            overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            posx, posy = posx * 512, posy * 512
            radius = 5
            # we can visualize per sub-query type
            # in violet the first type, and in green the second type
            if (i // the_constant) % 2 or not separated:
                color = (0, 255, 200)
            else:
                color = (255, 0, 200)
            draw.ellipse(
                (posx - radius, posy - radius, posx + radius, posy + radius),
                fill=color + (int(50 * attn.item()),)
            )
            result_image = Image.alpha_composite(result_image, overlay)
        return result_image

    def _visualize_pred_decoder_attention(self, predictions: list[Prediction], results: dict, layer: int = -1, separated = True):
        """
        Visualize the decoder attention.
        """
        # get the attention maps and the sampling locations
        attn_maps = results["decoder_attn_maps"][layer][0] # last decoder layer ; first image of batch
        # n * [B, Q, n_heads, n_levels, n_points, 2]
        sampling_locations = results["decoder_sampling_locations"][layer][0] # [Q, n_heads, n_levels, n_points, 2]
        Q, n_heads, n_levels, n_points, _ = sampling_locations.size()
        the_constant = n_heads * n_levels * n_points
        sampling_locations = sampling_locations.reshape(Q, -1, 2) # [Q * n_heads * n_levels * n_points, 2]
        attn_maps = attn_maps.reshape(Q, -1, 1) # [Q * n_heads * n_levels * num_points, 1]
        # create the images
        result_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        for prediction in predictions:
            if prediction.class_id != self.PAIR_CLASS_ID:
                continue
            query = 2 * prediction.query
            for i, (attn, (posx, posy)) in enumerate(zip(attn_maps[query], sampling_locations[query])):
                overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                posx, posy = posx * 512, posy * 512
                radius = 5
                # we can visualize per sub-query type
                # in violet the first type, and in green the second type
                color = (255, 0, 200)
                draw.ellipse(
                    (posx - radius, posy - radius, posx + radius, posy + radius),
                    fill=color + (int(150 * attn.item()),)
                )
                result_image = Image.alpha_composite(result_image, overlay)
            query = query + 1
            for i, (attn, (posx, posy)) in enumerate(zip(attn_maps[query], sampling_locations[query])):
                overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                posx, posy = posx * 512, posy * 512
                radius = 5
                # we can visualize per sub-query type
                # in violet the first type, and in green the second type
                color = (0, 255, 200)
                draw.ellipse(
                    (posx - radius, posy - radius, posx + radius, posy + radius),
                    fill=color + (int(50 * attn.item()),)
                )
                result_image = Image.alpha_composite(result_image, overlay)
        return result_image

    def _visualize_predictions(self, image: Image.Image, predictions: list[Prediction]) -> Image.Image:
        W, H = image.size
        result_image= image.copy().resize((512, 512)).convert("RGBA")
        blackboard = ImageDraw.Draw(result_image)
        for prediction in predictions:
            if prediction.class_id != self.PAIR_CLASS_ID:
                continue
            button_bbox = prediction.button_bbox
            counterpart_bbox = prediction.counterpart_bbox
            # first draw the button bbox
            x, y, w, h = button_bbox.cx / W * 512, button_bbox.cy / H * 512, button_bbox.w / W * 512, button_bbox.h / H * 512
            blackboard.rectangle(
                (x - w/2, y - h/2, x + w/2, y + h/2),
                outline=(0, 255, 0, 255),
                width=2
            )
            # then draw the fastener bbox
            x, y, w, h = counterpart_bbox.cx / W * 512, counterpart_bbox.cy / H * 512, counterpart_bbox.w / W * 512, counterpart_bbox.h / H * 512
            blackboard.rectangle(
                (x - w/2, y - h/2, x + w/2, y + h/2),
                outline=(0, 255, 255, 255),
                width=2
            )
        return result_image

    def _sampling_points_distance_pred(self, predictions: list[Prediction], results: dict):
        # n * [B, Q, n_heads, n_levels, n_points, 2]
        sampling_locations = results["decoder_sampling_locations"][-1][0] # [Q, n_heads, n_levels, n_points, 2]
        attn_maps = results["decoder_attn_maps"][-1][0] # [Q, n_heads, n_levels, n_points, 1]
        reference_points = results["reference_points"] # [Q, RpQ, 2]
        Q, n_heads, n_levels, n_points, _ = sampling_locations.size()
        # [Q, n_heads * n_levels * n_points, 2]
        sampling_locations = sampling_locations.reshape(Q, -1, 2)
        # [Q, n_heads * n_levels * n_points, 1]
        attn_maps = attn_maps.reshape(Q, -1, 1).squeeze(2)
        # [Q, n_heads * n_levels * n_points]
        button_distances = []
        counterpart_distances = []
        for prediction in predictions:
            if prediction.class_id != self.PAIR_CLASS_ID:
                continue
            button_bbox = prediction.button_bbox
            counterpart_bbox = prediction.counterpart_bbox
            b_bbox_center = torch.tensor([button_bbox.cx / 512, button_bbox.cy / 512])
            c_bbox_center = torch.tensor([counterpart_bbox.cx / 512, counterpart_bbox.cy / 512])
            # compute the center of mass for the buttons
            query = 2 * prediction.query
            points = sampling_locations[query] # buttons points only
            weighted = points * attn_maps[query][:, None]
            buttons_centers = weighted.sum(dim=0) / attn_maps[query].sum()
            # now for the counterpart
            query = 2 * prediction.query + 1
            points = sampling_locations[query] # counterparts points only
            weighted = points * attn_maps[query][:, None]
            counterpart_centers = weighted.sum(dim=0) / attn_maps[query].sum()
            # compute the distances
            button_dist = torch.linalg.vector_norm(buttons_centers - b_bbox_center)
            counterpart_dist = torch.linalg.vector_norm(counterpart_centers - c_bbox_center)
            button_distances.append(button_dist)
            counterpart_distances.append(counterpart_dist)
        m_button = np.mean(button_distances)
        m_counterparts = np.mean(counterpart_distances)
        print(
            f"Buttons: {m_button * 512:.2f}"
            f"\nCounterpart: {m_counterparts * 512: .2f}"
        )
