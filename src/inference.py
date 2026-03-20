"""
Inference engine for road damage detection.
High-level API that loads config, builds the ensemble, and exposes predict/visualize.
"""
from typing import List, Dict, Union, Optional
from pathlib import Path

import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ensemble import EnsembleDetector
from .wbf import CLASS_NAMES

# Mode → model indices mapping
MODE_INDICES = {
    "fast": [0],
    "balanced": [0, 1],
    "accurate": [0, 1, 2],
}

CLASS_LABELS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D50": "Other Corruption",
}

BOX_COLORS = {
    "D00": (239, 68, 68),
    "D10": (249, 115, 22),
    "D20": (234, 179, 8),
    "D40": (59, 130, 246),
    "D50": (139, 92, 246),
}


class RoadDamageDetector:
    """Main detector class for road damage detection."""

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        config_path: str = "config/ensemble_config.yaml",
        mode: str = "accurate",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_paths: Explicit list of model weight paths. If None, read from config.
            config_path: Path to ensemble_config.yaml.
            mode: Detection mode — 'fast', 'balanced', or 'accurate'.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.mode = mode

        # Load config
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Resolve device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Build model configs
        if model_paths is not None:
            cfg_models = self.config.get("models", [])
            model_configs = []
            for i, path in enumerate(model_paths):
                input_size = cfg_models[i]["input_size"] if i < len(cfg_models) else 640
                weight = cfg_models[i]["weight"] if i < len(cfg_models) else 1.0
                model_configs.append({"path": path, "input_size": input_size, "weight": weight})
        else:
            model_configs = [
                {"path": m["path"], "input_size": m["input_size"], "weight": m["weight"]}
                for m in self.config.get("models", [])
            ]

        wbf_params = self.config.get("wbf", {
            "iou_threshold": 0.5,
            "skip_box_threshold": 0.01,
            "conf_type": "avg",
        })

        self.ensemble = EnsembleDetector(model_configs, wbf_params, device=self.device)
        print(f"[RoadDamageDetector] Loaded {len(model_configs)} models on {self.device}, mode={self.mode}")

    def predict(
        self, image: Union[str, Image.Image], mode: Optional[str] = None, conf_threshold: float = 0.25
    ) -> List[Dict]:
        """
        Run inference on an image.

        Args:
            image: Path to image or PIL Image.
            mode: Override detection mode for this call.
            conf_threshold: Minimum confidence.

        Returns:
            List of detection dicts with keys: class, label, confidence, bbox.
        """
        active_mode = mode or self.mode
        model_indices = MODE_INDICES.get(active_mode, MODE_INDICES["accurate"])
        # Clamp to available models
        model_indices = [i for i in model_indices if i < len(self.ensemble.models)]
        if not model_indices:
            model_indices = [0]

        raw = self.ensemble.predict(image, model_indices=model_indices, conf_threshold=conf_threshold)

        detections = []
        for i in range(len(raw["scores"])):
            cls_idx = int(raw["labels"][i])
            cls_code = CLASS_NAMES.get(cls_idx, f"UNK{cls_idx}")
            x1, y1, x2, y2 = raw["boxes"][i].tolist()
            detections.append({
                "class": cls_code,
                "label": CLASS_LABELS.get(cls_code, cls_code),
                "confidence": float(raw["scores"][i]),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def visualize(self, image: Union[str, Image.Image], detections: List[Dict], save_path: Optional[str] = None) -> Image.Image:
        """Draw detections on the image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.copy().convert("RGB")

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_code = det["class"]
            color = BOX_COLORS.get(cls_code, (200, 200, 200))
            conf = det["confidence"]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_text = f'{cls_code} {conf:.0%}'
            draw.rectangle([x1, y1 - 18, x1 + len(label_text) * 8 + 8, y1], fill=color)
            draw.text((x1 + 4, y1 - 16), label_text, fill=(255, 255, 255), font=font)

        if save_path:
            image.save(save_path)

        return image
