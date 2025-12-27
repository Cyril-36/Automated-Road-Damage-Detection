# Quick Start Guide

Get started with Road Damage Detection in 5 minutes!

## Prerequisites

- Python 3.9+
- 8GB+ RAM
- GPU recommended (CUDA 11.8+)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/road-damage-detection.git
cd road-damage-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

**Option A**: Download from Google Drive
- Download models from [link]
- Place in `models/` directory

**Option B**: Train your own models
```bash
python src/train.py --config config/model1_config.yaml
```

## Basic Usage

### Run Web Interface

```bash
python deployment/gradio_app.py
```

Open your browser to `http://localhost:7860`

### Run REST API

```bash
python deployment/app.py
```

API available at `http://localhost:5000`

### Python Code

```python
from src.inference import RoadDamageDetector

# Initialize detector
detector = RoadDamageDetector(
    model_paths=[
        'models/yolov8n_640.pt',
        'models/yolov8s_640.pt',
        'models/yolov8s_1024.pt'
    ],
    mode='accurate'
)

# Detect damage
results = detector.predict('path/to/image.jpg')

# Visualize
detector.visualize(results, save_path='output.jpg')
```

## Detection Modes

- **Fast Mode**: ~20ms latency, 63.68% accuracy
- **Balanced Mode**: ~100ms latency, 65.05% accuracy
- **Accurate Mode**: ~150ms latency, 66.18% accuracy

## Next Steps

- Read the [full documentation](README.md)
- Check out the [training guide](TRAINING.md)
- Explore the [notebooks](../notebooks/)
- Try the [live demo](https://huggingface.co/spaces/Cyril-36/road-damage-detection)

## Troubleshooting

### Model not found
Make sure you've downloaded the model weights and placed them in the `models/` directory.

### CUDA out of memory
Reduce batch size or use CPU mode by setting `device='cpu'`.

### Import errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

## Support

- 📖 [Full Documentation](README.md)
- 🐛 [Issue Tracker](https://github.com/yourusername/road-damage-detection/issues)
- 💬 [Discussions](https://github.com/yourusername/road-damage-detection/discussions)
