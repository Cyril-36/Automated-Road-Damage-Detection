# Results

This directory contains the results from training and evaluation of the road damage detection models.

## Directory Structure

```
results/
├── yolov8n_640/          # YOLOv8n @ 640×640 results
├── yolov8s_640/          # YOLOv8s @ 640×640 results
├── yolov8s_1024/         # YOLOv8s @ 1024×1024 results
├── ensemble/             # Ensemble and TTA results
├── figures/              # Key figures for README
└── metrics/              # Evaluation metrics (CSV files)
```

## Model Results

### YOLOv8n @ 640×640 (Model 1)
- **mAP@50**: 60.01%
- **Parameters**: 3.2M
- **Inference Time**: ~15ms (GPU)
- **Use Case**: Real-time, resource-constrained

Files in `yolov8n_640/`:
- `confusion_matrix.png` - Confusion matrix visualization
- `results.png` - Training curves
- `results.csv` - Detailed metrics per epoch

### YOLOv8s @ 640×640 (Model 2)
- **mAP@50**: 63.43%
- **Parameters**: 11.2M
- **Inference Time**: ~20ms (GPU)
- **Use Case**: Balanced speed/accuracy

Files in `yolov8s_640/`:
- `confusion_matrix.png` - Confusion matrix visualization
- `training_curves.png` - Training progress
- `results.csv` - Detailed metrics per epoch
- `test_results.json` - Test set evaluation

### YOLOv8s @ 1024×1024 (Model 3)
- **mAP@50**: 63.68%
- **Parameters**: 11.2M
- **Inference Time**: ~50ms (GPU)
- **Use Case**: High detail, accuracy-critical

Files in `yolov8s_1024/`:
- `confusion_matrix.png` - Confusion matrix visualization
- `training_curves.png` - Training progress
- `results.csv` - Detailed metrics per epoch
- `test_results.json` - Test set evaluation

### Ensemble Results
- **mAP@50**: 66.18% ⭐
- **Fusion Method**: Weighted Boxes Fusion
- **Weights**: [1.0, 1.5, 2.0]

Files in `ensemble/`:
- `tta_ensemble_comparison.csv` - Comparison with TTA
- `tta_ensemble_results.csv` - Detailed ensemble results
- `tta_ensemble_comparison.png` - Visual comparison

## Figures

The `figures/` directory contains key visualizations used in the main README:
- `confusion_matrix.png` - Best model confusion matrix
- `training_curves.png` - Training progress
- `pr_curve.png` - Precision-Recall curve
- `performance_comparison.png` - Model comparison chart

## Metrics

The `metrics/` directory contains CSV files with detailed evaluation metrics:
- `map_50.csv` - mAP@50 scores
- `precision.csv` - Precision scores
- `recall.csv` - Recall scores
- `f1_score.csv` - F1 scores
- `per_class_comparison.csv` - Per-class performance
- `overall_comparison.csv` - Overall model comparison
