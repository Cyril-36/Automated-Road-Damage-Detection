# 🛣️ Automated Road Damage Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green?logo=ultralytics&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![mAP](https://img.shields.io/badge/Ensemble_mAP@50-66.18%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)

**Multi-Resolution YOLOv8 Ensemble for Automated Road Damage Detection**

[Overview](#overview) • [Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Results](#results) • [Architecture](#architecture) • [Documentation](#documentation)

---

## ℹ️ Overview

This project implements a cutting-edge **multi-resolution YOLOv8 ensemble system** for automated road damage detection. By leveraging multiple YOLOv8 models trained at different resolutions and fused using **Weighted Boxes Fusion (WBF)**, the system achieves state-of-the-art performance on the RDD2022 dataset with an **ensemble mAP@50 of 66.18%**.

The system is designed for real-world deployment in infrastructure monitoring, automated damage assessment, and road maintenance planning. It includes both training pipelines and production-ready inference interfaces (Flask API and Gradio UI).

---

## ⚡ Features

- **Multi-Resolution Ensemble Architecture**: Combines YOLOv8 models at different input resolutions for optimal accuracy
  - YOLOv8n @ 640×640: Real-time inference (~15ms)
  - YOLOv8s @ 640×640: Balanced performance (~20ms)
  - YOLOv8s @ 1024×1024: High-accuracy detection (~50ms)

- **Advanced Fusion Strategy**: Weighted Boxes Fusion (WBF) intelligently combines predictions from all models

- **Comprehensive Training Pipeline**:
  - EDA and data analysis notebooks
  - Hyperparameter tuning workflows
  - Training scripts with configurable parameters
  - Model evaluation and benchmarking

- **Production Deployment Options**:
  - RESTful Flask API for backend integration
  - Interactive Gradio web interface for manual testing
  - Containerized deployment support

- **Robust Inference Engine**:
  - Batch processing capabilities
  - Configurable confidence thresholds
  - Multi-format output (JSON, images with annotations)
  - GPU and CPU support

---

## 📊 Performance Results

| Model | Resolution | mAP@50 | Parameters | Inference Time |
|-------|-----------|--------|-----------|----------------|
| YOLOv8n | 640×640 | 60.01% | 3.2M | ~15ms |
| YOLOv8s | 640×640 | 63.43% | 11.2M | ~20ms |
| YOLOv8s | 1024×1024 | 63.68% | 11.2M | ~50ms |
| **Ensemble (WBF)** | **Multi** | **66.18%** ⭐ | - | ~85ms |

**Key Metrics (Ensemble)**:
- **Precision**: High confidence in predictions
- **Recall**: Robust detection of various damage types
- **F1-Score**: Balanced performance across all classes
- Per-class performance available in `results/metrics/`

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (recommended for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for training)
- Git

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/Cyril-36/Automated-Road-Damage-Detection.git
cd Automated-Road-Damage-Detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependency Overview

**Core Libraries**:
- `torch>=2.0.0`: Deep learning framework
- `ultralytics>=8.0.0`: YOLOv8 implementation
- `opencv-python>=4.8.0`: Computer vision operations

**Ensemble Fusion**:
- `ensemble-boxes>=1.0.9`: Weighted Boxes Fusion

**Utilities**:
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`: Data processing and visualization
- `Flask`, `Flask-CORS`, `gradio`: Web interfaces
- `albumentations`: Data augmentation

---

## 🚀 Quick Start

### 1. Run Inference on a Single Image

```python
from src.inference import RoadDamageDetector

# Initialize the detector
detector = RoadDamageDetector(
    model_paths=[
        'models/yolov8n_640.pt',
        'models/yolov8s_640.pt',
        'models/yolov8s_1024.pt'
    ],
    conf_threshold=0.5
)

# Run detection
results = detector.detect('path/to/image.jpg')

# Visualize results
detector.visualize(results, save_path='output.jpg')
```

### 2. Batch Processing

```python
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = detector.detect_batch(images)
detector.export_results(results, format='json', output_path='results.json')
```

### 3. Web Interface (Gradio)

```bash
python deployment/gradio_app.py
# Visit http://localhost:7860 in your browser
```

### 4. REST API (Flask)

```bash
python deployment/app.py
# API available at http://localhost:5000

# Example request:
curl -X POST -F "image=@image.jpg" http://localhost:5000/predict
```

---

## 📁 Project Structure

```
Automated-Road-Damage-Detection/
├── Notebooks/                  # Jupyter notebooks for analysis and training
│   ├── EDA-RDD.ipynb
│   ├── baseline-rdd2022.ipynb
│   ├── TuningRDD22.ipynb
│   └── Model_Evaluation.ipynb
├── src/                        # Source code
│   ├── inference.py
│   ├── ensemble.py
│   └── wbf.py
├── deployment/                 # Production deployment
│   ├── app.py
│   └── gradio_app.py
├── models/                     # Trained model weights
│   ├── yolov8n_640.pt
│   ├── yolov8s_640.pt
│   └── yolov8s_1024.pt
├── config/                     # Configuration files
│   └── model_config.yaml
├── data/                       # Dataset directory
│   └── RDD2022/
├── docs/                       # Additional documentation
├── scripts/                    # Utility scripts
├── results/                    # Training and evaluation results
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📥 Dataset Setup

This project uses the **RDD2022 Road Damage Detection 2022** dataset.

### Download Instructions

1. Download from the [official RDD2022 repository](https://github.com/sekilab/RDD2022)
2. Extract and organize into the project:

```bash
mkdir -p data/RDD2022/{train,test}/{images,labels}
cp -r /path/to/RDD2022/train/images data/RDD2022/train/
cp -r /path/to/RDD2022/train/labels data/RDD2022/train/
cp -r /path/to/RDD2022/test/images data/RDD2022/test/
cp -r /path/to/RDD2022/test/labels data/RDD2022/test/
```

---

## 🔧 Training

```bash
python scripts/train.py --config config/model_config.yaml
```

### Hyperparameter Configuration

Edit `config/model_config.yaml`:

```yaml
models:
  yolov8n:
    resolution: 640
    epochs: 100
    batch_size: 16
    learning_rate: 0.001
  yolov8s:
    resolution: 640
    epochs: 100
    batch_size: 16
    learning_rate: 0.001
  yolov8s_large:
    resolution: 1024
    epochs: 100
    batch_size: 8
    learning_rate: 0.0005

ensemble:
  weights: [1.0, 1.5, 2.0]
  iou_threshold: 0.5
  confidence_threshold: 0.5
```

---

## 🏗️ Architecture

### System Pipeline

```
INPUT IMAGE
    ↓
    ├─→ YOLOv8n (640×640) → Predictions 1
    ├─→ YOLOv8s (640×640) → Predictions 2
    └─→ YOLOv8s (1024×1024) → Predictions 3
    ↓
    Weighted Boxes Fusion (WBF)
    ↓
    FINAL ENSEMBLE PREDICTIONS
    ↓
    Output: Bounding Boxes + Classes + Confidence Scores
```

### Weighted Boxes Fusion Algorithm

1. **Cluster overlapping boxes** (IoU threshold: 0.5)
2. **Calculate weighted coordinates**: `(b1*w1 + b2*w2 + b3*w3) / (w1 + w2 + w3)`
3. **Average confidence**: `(c1 + c2 + c3) / 3`
4. **Select ensemble class** from majority vote

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t road-damage-detector .

# Run container
docker run -p 5000:5000 road-damage-detector
```

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "deployment/app.py"]
```

---

## 📈 Results & Evaluation

### Per-Class Performance

Detailed metrics for each damage class:
- D00: Longitudinal Cracks
- D10: Transverse Cracks
- D20: Alligator Cracks
- D40: Potholes
- D50: White Paint Lines

See `results/metrics/` for detailed analysis including:
- Confusion matrices
- Precision-Recall curves
- Training curves
- Per-class comparisons

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Make your changes and commit: `git commit -am 'Add improvement'`
4. Push to the branch: `git push origin feature/improvement`
5. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [RDD2022 Dataset](https://github.com/sekilab/RDD2022)
- [Weighted Boxes Fusion Paper](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## 🙏 Acknowledgments

- **Dataset**: Thanks to the RDD2022 dataset creators
- **Framework**: Ultralytics team for the excellent YOLOv8 implementation
- **Community**: All contributors who support this project

---

## 📞 Support & Contact

- **Issues**: [Open an issue on GitHub](https://github.com/Cyril-36/Automated-Road-Damage-Detection/issues)
- **Email**: cyrilchaitanya@gmail.com
- **Documentation**: Check the `docs/` folder for detailed guides

---

## ✨ Roadmap

- [ ] Add real-time video stream processing
- [ ] Implement MLOps pipeline (DVC, CML)
- [ ] Add model quantization for edge deployment
- [ ] Integrate with cloud platforms (AWS, GCP, Azure)
- [ ] Create mobile app for road assessment
- [ ] Add damage severity classification

---

**Made with ❤️ by Cyril Chaitanya Pudota**

⭐ If this project helps you, please consider giving it a star!

[Back to Top](#-automated-road-damage-detection)
