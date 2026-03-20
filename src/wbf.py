"""
Weighted Boxes Fusion implementation
Uses the ensemble_boxes library for robust box fusion.
"""
import numpy as np
from ensemble_boxes import weighted_boxes_fusion


CLASS_NAMES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "D50",
}


class WeightedBoxesFusion:
    """Weighted Boxes Fusion for combining predictions from multiple models."""

    def __init__(self, weights, iou_threshold=0.5, skip_box_threshold=0.01, conf_type="avg"):
        self.weights = weights
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_type = conf_type

    def fuse(self, predictions):
        """
        Fuse predictions from multiple models.

        Args:
            predictions: list of dicts, one per model. Each dict has:
                - boxes: np.ndarray (N, 4) in [x1, y1, x2, y2] normalised 0-1
                - scores: np.ndarray (N,)
                - labels: np.ndarray (N,) int class indices

        Returns:
            dict with fused boxes, scores, labels (all np arrays)
        """
        if not predictions:
            return {"boxes": np.empty((0, 4)), "scores": np.empty(0), "labels": np.empty(0, dtype=int)}

        boxes_list = [p["boxes"] for p in predictions]
        scores_list = [p["scores"] for p in predictions]
        labels_list = [p["labels"] for p in predictions]

        weights = self.weights[: len(predictions)]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=self.iou_threshold,
            skip_box_thr=self.skip_box_threshold,
            conf_type=self.conf_type,
        )

        return {
            "boxes": fused_boxes,
            "scores": fused_scores,
            "labels": fused_labels.astype(int),
        }
