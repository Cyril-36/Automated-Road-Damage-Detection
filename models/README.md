# Model Weights

This directory contains the trained YOLOv8 model weights for road damage detection.

## Download Pretrained Models

The model weights are too large to be stored in GitHub. Please download them from one of the following sources:

### Option 1: Google Drive
Download the pretrained models from Google Drive:
- [YOLOv8n @ 640×640](https://drive.google.com/...) - 6.5 MB
- [YOLOv8s @ 640×640](https://drive.google.com/...) - 22 MB
- [YOLOv8s @ 1024×1024](https://drive.google.com/...) - 22 MB

### Option 2: HuggingFace
```bash
# Install huggingface_hub
pip install huggingface_hub

# Download models
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Cyril-36/road-damage-detection', filename='yolov8n_640.pt', local_dir='models/')
hf_hub_download(repo_id='Cyril-36/road-damage-detection', filename='yolov8s_640.pt', local_dir='models/')
hf_hub_download(repo_id='Cyril-36/road-damage-detection', filename='yolov8s_1024.pt', local_dir='models/')
"
```

### Option 3: Direct Links
If you have the models hosted elsewhere, add direct download links here.

## Expected Directory Structure

After downloading, your `models/` directory should look like this:

```
models/
├── README.md (this file)
├── yolov8n_640.pt
├── yolov8s_640.pt
└── yolov8s_1024.pt
```

## Model Details

| Model | Input Size | Parameters | mAP@50 | Inference Time (GPU) | Use Case |
|-------|-----------|------------|--------|---------------------|----------|
| YOLOv8n@640 | 640×640 | 3.2M | 60.01% | ~15ms | Real-time, resource-constrained |
| YOLOv8s@640 | 640×640 | 11.2M | 63.43% | ~20ms | Balanced speed/accuracy |
| YOLOv8s@1024 | 1024×1024 | 11.2M | 63.68% | ~50ms | High detail, accuracy-critical |

## Training Information

All models were trained on the RDD2022 dataset with the following configurations:

- **Dataset**: RDD2022 (30,708 training images)
- **Augmentation**: HSV, rotation, translation, scaling, horizontal flip, mosaic
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 with cosine annealing
- **Training Hardware**: NVIDIA GPU (Tesla T4 or better)

For detailed training configurations, see the `config/` directory.

## Notes

- Model files are excluded from git tracking via `.gitignore` due to file size
- If you train your own models, place them in this directory
- Ensure model filenames match the configuration files
