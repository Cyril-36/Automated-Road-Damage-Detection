"""
Ensemble detection logic — runs multiple YOLOv8 models and fuses with WBF.
"""
import numpy as np
from ultralytics import YOLO
from PIL import Image
from .wbf import WeightedBoxesFusion


class EnsembleDetector:
    """Ensemble detector combining multiple YOLOv8 models."""

    def __init__(self, model_configs, wbf_params, device="cpu"):
        """
        Args:
            model_configs: list of dicts with keys: path, input_size, weight
            wbf_params: dict with iou_threshold, skip_box_threshold, conf_type
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.models = []
        self.input_sizes = []
        weights = []

        for cfg in model_configs:
            model = YOLO(cfg["path"])
            model.to(device)
            self.models.append(model)
            self.input_sizes.append(cfg["input_size"])
            weights.append(cfg["weight"])

        self.fusion = WeightedBoxesFusion(
            weights=weights,
            iou_threshold=wbf_params.get("iou_threshold", 0.5),
            skip_box_threshold=wbf_params.get("skip_box_threshold", 0.01),
            conf_type=wbf_params.get("conf_type", "avg"),
        )

    def _run_single_model(self, model, image, input_size, conf_threshold):
        """Run a single YOLO model and return normalised predictions."""
        results = model.predict(
            source=image,
            imgsz=input_size,
            conf=conf_threshold,
            verbose=False,
        )
        result = results[0]
        img_h, img_w = result.orig_shape

        if result.boxes is None or len(result.boxes) == 0:
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty(0),
                "labels": np.empty(0, dtype=int),
            }

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        # Normalise boxes to 0-1
        xyxy[:, [0, 2]] /= img_w
        xyxy[:, [1, 3]] /= img_h
        xyxy = np.clip(xyxy, 0, 1)

        return {"boxes": xyxy, "scores": scores, "labels": labels}

    def predict(self, image, model_indices=None, conf_threshold=0.25):
        """
        Run ensemble prediction.

        Args:
            image: PIL Image or path string
            model_indices: which models to use (None = all)
            conf_threshold: confidence threshold

        Returns:
            dict with boxes (absolute xyxy), scores, labels, img_size
        """
        if isinstance(image, str):
            image = Image.open(image)

        img_w, img_h = image.size
        indices = model_indices if model_indices is not None else list(range(len(self.models)))

        if len(indices) == 1:
            # Single model — no fusion needed
            pred = self._run_single_model(
                self.models[indices[0]], image, self.input_sizes[indices[0]], conf_threshold
            )
            boxes_abs = pred["boxes"].copy()
            boxes_abs[:, [0, 2]] *= img_w
            boxes_abs[:, [1, 3]] *= img_h
            return {
                "boxes": boxes_abs,
                "scores": pred["scores"],
                "labels": pred["labels"],
                "img_size": (img_w, img_h),
            }

        # Multi-model: collect predictions then fuse
        preds = []
        for idx in indices:
            p = self._run_single_model(
                self.models[idx], image, self.input_sizes[idx], conf_threshold
            )
            preds.append(p)

        fused = self.fusion.fuse(preds)

        # Filter by confidence
        mask = fused["scores"] >= conf_threshold
        boxes_norm = fused["boxes"][mask]
        scores = fused["scores"][mask]
        labels = fused["labels"][mask]

        # Convert back to absolute coords
        boxes_abs = boxes_norm.copy()
        boxes_abs[:, [0, 2]] *= img_w
        boxes_abs[:, [1, 3]] *= img_h

        return {
            "boxes": boxes_abs,
            "scores": scores,
            "labels": labels,
            "img_size": (img_w, img_h),
        }
