# 🛣️ Automated Road Damage Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green?logo=ultralytics&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![mAP](https://img.shields.io/badge/Ensemble_mAP@50-66.18%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)

**Multi-Resolution YOLOv8 Ensemble for Automated Road Damage Detection**

[Overview](#overview) • [Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Results](#results) • [Architecture](#architecture) • [Documentation](#documentation)

</div>

---

## 📋 Overview

This project implements a cutting-edge **multi-resolution YOLOv8 ensemble system** for automated road damage detection. By leveraging multiple YOLOv8 models trained at different resolutions and fused using **Weighted Boxes Fusion (WBF)**, the system achieves state-of-the-art performance on the RDD2022 dataset with an **ensemble mAP@50 of 66.18%**.

The system is designed for real-world deployment in infrastructure monitoring, automated damage assessment, and road maintenance planning. It includes both training pipelines and production-ready inference interfaces (Flask API and Gradio UI).

---

## ✨ Features

- **Multi-Resolution Ensemble Architecture**: Combines YOLOv8 models at different input resolutions for optimal accuracy
-   - YOLOv8n @ 640×640: Real-time inference (~15ms)
    -   - YOLOv8s @ 640×640: Balanced performance (~20ms)
        -   - YOLOv8s @ 1024×1024: High-accuracy detection (~50ms)
         
            - - **Advanced Fusion Strategy**: Weighted Boxes Fusion (WBF) intelligently combines predictions from all models
             
              - - **Comprehensive Training Pipeline**:
                -   - EDA and data analysis notebooks
                    -   - Hyperparameter tuning workflows
                        -   - Training scripts with configurable parameters
                            -   - Model evaluation and benchmarking
                             
                                - - **Production Deployment Options**:
                                  -   - RESTful Flask API for backend integration
                                      -   - Interactive Gradio web interface for manual testing
                                          -   - Containerized deployment support
                                           
                                              - - **Robust Inference Engine**:
                                                -   - Batch processing capabilities
                                                    -   - Configurable confidence thresholds
                                                        -   - Multi-format output (JSON, images with annotations)
                                                            -   - GPU and CPU support
                                                             
                                                                - ---

                                                                ## 📊 Performance Results

                                                                | Model | Resolution | mAP@50 | Parameters | Inference Time |
                                                                |-------|-----------|--------|-----------|-----------------|
                                                                | YOLOv8n | 640×640 | 60.01% | 3.2M | ~15ms |
                                                                | YOLOv8s | 640×640 | 63.43% | 11.2M | ~20ms |
                                                                | YOLOv8s | 1024×1024 | 63.68% | 11.2M | ~50ms |
                                                                | **Ensemble (WBF)** | **Multi** | **66.18%** ⭐ | - | ~85ms |

                                                                **Key Metrics (Ensemble)**:
                                                                - Precision: High confidence in predictions
                                                                - - Recall: Robust detection of various damage types
                                                                  - - F1-Score: Balanced performance across all classes
                                                                    - - Per-class performance available in `results/metrics/`
                                                                     
                                                                      - ---

                                                                      ## 🚀 Installation

                                                                      ### Prerequisites

                                                                      - Python 3.9 or higher
                                                                      - - CUDA 11.8+ (recommended for GPU acceleration)
                                                                        - - 8GB+ RAM (16GB+ recommended for training)
                                                                          - - Git
                                                                           
                                                                            - ### Setup Steps
                                                                           
                                                                            - ```bash
                                                                              # 1. Clone the repository
                                                                              git clone https://github.com/Cyril-36/Automated-Road-Damage-Detection.git
                                                                              cd Automated-Road-Damage-Detection

                                                                              # 2. Create virtual environment
                                                                              python -m venv venv
                                                                              source venv/bin/activate  # On Windows: venv\Scripts\activate

                                                                              # 3. Install dependencies
                                                                              pip install -r requirements.txt

                                                                              # 4. (Optional) For CUDA support
                                                                              # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                                                                              ```

                                                                              ### Dependency Overview

                                                                              **Core Libraries**:
                                                                              - `torch>=2.0.0` & `torchvision>=0.15.0` - Deep learning framework
                                                                              - - `ultralytics>=8.0.0` - YOLOv8 implementation
                                                                                - - `opencv-python>=4.8.0` - Computer vision operations
                                                                                 
                                                                                  - **Ensemble & Fusion**:
                                                                                  - - `ensemble-boxes>=1.0.9` - Weighted Boxes Fusion
                                                                                   
                                                                                    - **Utilities**:
                                                                                    - - `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly` - Data processing and visualization
                                                                                      - - `Flask`, `Flask-CORS`, `gradio` - Web interfaces
                                                                                        - - `albumentations` - Data augmentation
                                                                                         
                                                                                          - ---

                                                                                          ## 🎯 Quick Start

                                                                                          ### 1. Run Inference on a Single Image

                                                                                          ```python
                                                                                          from src.inference import RoadDamageDetector

                                                                                          # Initialize the detector
                                                                                          detector = RoadDamageDetector(
                                                                                              model_paths=['models/yolov8n_640.pt', 'models/yolov8s_640.pt', 'models/yolov8s_1024.pt'],
                                                                                              conf_threshold=0.5
                                                                                          )

                                                                                          # Run detection
                                                                                          results = detector.detect('path/to/image.jpg')

                                                                                          # Visualize results
                                                                                          detector.visualize_results(results, save_path='output.jpg')
                                                                                          ```

                                                                                          ### 2. Batch Processing

                                                                                          ```python
                                                                                          # Process multiple images
                                                                                          images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
                                                                                          results = detector.detect_batch(images)

                                                                                          # Export results to JSON
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
                                                                                          # Server runs on http://localhost:5000

                                                                                          # Example request:
                                                                                          # curl -X POST -F "image=@image.jpg" http://localhost:5000/predict
                                                                                          ```

                                                                                          ---

                                                                                          ## 🎨 UI/UX Wireframes

                                                                                          ### Gradio Interface Mockup

                                                                                          ```
                                                                                          ╔════════════════════════════════════════════════════════════════════════════╗
                                                                                          ║                   🛣️  Road Damage Detection System                        ║
                                                                                          ╠════════════════════════════════════════════════════════════════════════════╣
                                                                                          ║                                                                            ║
                                                                                          ║  ┌─────────────────────────────────────┬────────────────────────────────┐ ║
                                                                                          ║  │                                     │                                │ ║
                                                                                          ║  │      Input Image Upload Area        │     Confidence Threshold       │ ║
                                                                                          ║  │                                     │                                │ ║
                                                                                          ║  │   ┌─────────────────────────────┐   │  Slider: ━━━●━━━ 0.50         │ ║
                                                                                          ║  │   │  📸 Drop Image or Click     │   │                                │ ║
                                                                                          ║  │   │  to Upload                  │   │   ┌──────────────────────────┐ │ ║
                                                                                          ║  │   │  (JPG, PNG supported)       │   │   │ Model Selection:         │ │ ║
                                                                                          ║  │   │                             │   │   │ ✓ YOLOv8n 640×640        │ │ ║
                                                                                          ║  │   │                             │   │   │ ✓ YOLOv8s 640×640        │ │ ║
                                                                                          ║  │   └─────────────────────────────┘   │   │ ✓ YOLOv8s 1024×1024      │ │ ║
                                                                                          ║  │                                     │   │ ✓ Ensemble (WBF)          │ │ ║
                                                                                          ║  │   ┌──────────────────────────────┐  │   │                          │ │ ║
                                                                                          ║  │   │      [🚀 Detect Damage]      │  │   │ ✓ Show Annotations       │ │ ║
                                                                                          ║  │   │      [📊 Batch Process]      │  │   │ ✓ Export JSON            │ │ ║
                                                                                          ║  │   └──────────────────────────────┘  │   └──────────────────────────┘ │ ║
                                                                                          ║  │                                     │                                │ ║
                                                                                          ║  └─────────────────────────────────────┴────────────────────────────────┘ ║
                                                                                          ║                                                                            ║
                                                                                          ║  ┌──────────────────────────────────────────────────────────────────────┐ ║
                                                                                          ║  │                         Output Results                              │ ║
                                                                                          ║  ├──────────────────────────────────────────────────────────────────────┤ ║
                                                                                          ║  │                                                                      │ ║
                                                                                          ║  │  ┌────────────────┐  Detection Time: 85ms  |  Detections Found: 5  │ ║
                                                                                          ║  │  │                │  mAP Score: 66.18%     |  Confidence: 92.3%    │ ║
                                                                                          ║  │  │                │                                                 │ ║
                                                                                          ║  │  │  Annotated     │  ┌──────────────────────────────────────────┐  │ ║
                                                                                          ║  │  │  Output        │  │ Detection Results:                       │  │ ║
                                                                                          ║  │  │  Image         │  │ • D00: Longitudinal Cracks - 94.2%      │  │ ║
                                                                                          ║  │  │                │  │ • D10: Transverse Cracks - 89.7%        │  │ ║
                                                                                          ║  │  │                │  │ • D20: Aligator Cracks - 91.5%          │  │ ║
                                                                                          ║  │  │                │  │ • D40: Potholes - 87.3%                 │  │ ║
                                                                                          ║  │  └────────────────┘  │ • D50: White Paint Lines - 93.1%        │  │ ║
                                                                                          ║  │                      └──────────────────────────────────────────┘  │ ║
                                                                                          ║  │                                                                      │ ║
                                                                                          ║  │  [📥 Download Image] [📋 Copy JSON] [🔄 Clear] [💾 Save Report]   │ ║
                                                                                          ║  │                                                                      │ ║
                                                                                          ║  └──────────────────────────────────────────────────────────────────────┘ ║
                                                                                          ║                                                                            ║
                                                                                          ╚════════════════════════════════════════════════════════════════════════════╝
                                                                                          ```

                                                                                          ### Flask REST API Endpoints Structure

                                                                                          ```
                                                                                          ╔═══════════════════════════════════════════════════════════════════════════╗
                                                                                          ║                      Flask REST API Architecture                         ║
                                                                                          ╠═══════════════════════════════════════════════════════════════════════════╣
                                                                                          ║                                                                           ║
                                                                                          ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
                                                                                          ║  │                         API Gateway                                 │ ║
                                                                                          ║  │                     (http://localhost:5000)                         │ ║
                                                                                          ║  └─────────────────────────────────────────────────────────────────────┘ ║
                                                                                          ║           ↓                    ↓                      ↓                   ║
                                                                                          ║  ┌────────────────┐  ┌─────────────────┐  ┌───────────────────────┐    ║
                                                                                          ║  │  POST /predict │  │  POST /batch    │  │ GET /health           │    ║
                                                                                          ║  ├────────────────┤  ├─────────────────┤  ├───────────────────────┤    ║
                                                                                          ║  │ Input:         │  │ Input:          │  │ Status: API Running   │    ║
                                                                                          ║  │ - Image (file) │  │ - Images (list) │  │ Models: 3 Loaded      │    ║
                                                                                          ║  │ - conf_th: 0.5 │  │ - conf_th: 0.5  │  │ GPU: Available        │    ║
                                                                                          ║  │ - use_ensemble │  │ - use_ensemble  │  │ Uptime: 12h 34m       │    ║
                                                                                          ║  │                │  │                 │  │                       │    ║
                                                                                          ║  │ Output:        │  │ Output:         │  └───────────────────────┘    ║
                                                                                          ║  │ {              │  │ [{              │                                ║
                                                                                          ║  │   image_id: 1  │  │   image_id: 1,  │  ┌───────────────────────┐    ║
                                                                                          ║  │   detections:  │  │   detections: [│  │ GET /models           │    ║
                                                                                          ║  │   [            │  │   {             │  ├───────────────────────┤    ║
                                                                                          ║  │     {          │  │     class: D00, │  │ Available Models:     │    ║
                                                                                          ║  │       class: 1 │  │     conf: 0.94, │  │ - yolov8n_640         │    ║
                                                                                          ║  │       conf:0.94│  │     bbox: [...]│  │ - yolov8s_640         │    ║
                                                                                          ║  │       bbox:[..│  │   }, ...        │  │ - yolov8s_1024        │    ║
                                                                                          ║  │     }, ...     │  │   ]             │  │ - ensemble_wbf        │    ║
                                                                                          ║  │   ],           │  │ }, ...]         │  │                       │    ║
                                                                                          ║  │   time_ms: 85  │  │ time_ms: 180    │  └───────────────────────┘    ║
                                                                                          ║  │ }              │  │ }               │                                ║
                                                                                          ║  └────────────────┘  └─────────────────┘  ┌───────────────────────┐    ║
                                                                                          ║                                            │ POST /train           │    ║
                                                                                          ║                                            ├───────────────────────┤    ║
                                                                                          ║  ┌────────────────┐  ┌─────────────────┐  │ Input:                │    ║
                                                                                          ║  │ GET /history   │  │ DELETE /cache   │  │ - config (JSON)       │    ║
                                                                                          ║  ├────────────────┤  ├─────────────────┤  │ - dataset_path        │    ║
                                                                                          ║  │ Returns:       │  │ Status: Success │  │                       │    ║
                                                                                          ║  │ - Last 10      │  │ Freed: 2.3 GB   │  │ Output:               │    ║
                                                                                          ║  │   predictions  │  │ Available: 8 GB │  │ - training_id         │    ║
                                                                                          ║  │ - Timestamps   │  │                 │  │ - status: queued      │    ║
                                                                                          ║  │ - Metrics      │  │                 │  │ - eta: 4h 20m         │    ║
                                                                                          ║  │                │  │                 │  │                       │    ║
                                                                                          ║  └────────────────┘  └─────────────────┘  └───────────────────────┘    ║
                                                                                          ║                                                                           ║
                                                                                          ╚═══════════════════════════════════════════════════════════════════════════╝
                                                                                          ```

                                                                                          ---

                                                                                          ## 🏗️ System Architecture

                                                                                          ### Overall Pipeline Architecture

                                                                                          ```
                                                                                          ┌──────────────────────────────────────────────────────────────────────────┐
                                                                                          │                         Road Damage Detection Pipeline                   │
                                                                                          └──────────────────────────────────────────────────────────────────────────┘

                                                                                                                        INPUT IMAGE
                                                                                                                            │
                                                                                                              ┌─────────────┴──────────────┐
                                                                                                              ▼                            ▼
                                                                                                      ┌──────────────────┐      ┌──────────────────┐
                                                                                                      │  YOLOv8n 640×640 │      │  YOLOv8s 640×640 │
                                                                                                      │  (3.2M params)   │      │  (11.2M params)  │
                                                                                                      │ mAP@50: 60.01%   │      │ mAP@50: 63.43%   │
                                                                                                      └──────┬───────────┘      └────────┬─────────┘
                                                                                                             │                           │
                                                                                                             │  Predictions [1]          │  Predictions [2]
                                                                                                             │                           │
                                                                                                             └─────────────┬─────────────┘
                                                                                                                           │
                                                                                                                           ▼
                                                                                                              ┌──────────────────────┐
                                                                                                              │ YOLOv8s 1024×1024    │
                                                                                                              │ (11.2M params)       │
                                                                                                              │ mAP@50: 63.68%       │
                                                                                                              └──────┬───────────────┘
                                                                                                                     │
                                                                                                                     │ Predictions [3]
                                                                                                                     │
                                                                                                      ┌──────────────┴──────────────┐
                                                                                                      │                             │
                                                                                                      ▼                             ▼
                                                                                                All Predictions ────────────────────────
                                                                                                                    │
                                                                                                                    ▼
                                                                                                      ┌──────────────────────────────────┐
                                                                                                      │  Weighted Boxes Fusion (WBF)     │
                                                                                                      │  Weights: [1.0, 1.5, 2.0]        │
                                                                                                      │  IoU Threshold: 0.5              │
                                                                                                      └──────────────────┬───────────────┘
                                                                                                                         │
                                                                                                          ┌──────────────┴──────────────┐
                                                                                                          │                             │
                                                                                                          ▼                             ▼
                                                                                                      ENSEMBLE                    FINAL PREDICTIONS
                                                                                                      mAP@50: 66.18% ⭐          • Bounding Boxes
                                                                                                                                 • Class Labels
                                                                                                                                 • Confidence Scores
                                                                                                                                 • Damage Classification
                                                                                          ```

                                                                                          ### Model Fusion Strategy

                                                                                          ```
                                                                                          Individual Model Outputs:
                                                                                          ────────────────────────

                                                                                          Model 1 (YOLOv8n@640):         Model 2 (YOLOv8s@640):         Model 3 (YOLOv8s@1024):
                                                                                          ┌────────────────────┐         ┌────────────────────┐         ┌────────────────────┐
                                                                                          │ Box: [100,100,200  │         │ Box: [102,98,198   │         │ Box: [99,101,201   │
                                                                                          │ Class: D00         │         │ Class: D00         │         │ Class: D00         │
                                                                                          │ Conf: 0.92         │         │ Conf: 0.96         │         │ Conf: 0.94         │
                                                                                          └────────────────────┘         └────────────────────┘         └────────────────────┘

                                                                                                              Weighted Boxes Fusion Algorithm:
                                                                                                              ─────────────────────────────

                                                                                                              1. Cluster overlapping boxes (IoU > 0.5)
                                                                                                              2. Calculate weighted coordinates:
                                                                                                                 box_coords = (b1*w1 + b2*w2 + b3*w3) / (w1+w2+w3)
                                                                                                              3. Average confidence: conf = (c1 + c2 + c3) / 3
                                                                                                              4. Select ensemble class from majority vote

                                                                                                                      ▼

                                                                                                              ┌────────────────────┐
                                                                                                              │ Fused Detection:   │
                                                                                                              │ Box: [100.3,99.7   │
                                                                                                              │       200.3,199.7] │ ← More precise box
                                                                                                              │ Class: D00         │
                                                                                                              │ Conf: 0.94 (avg)   │ ← Balanced confidence
                                                                                                              │ Method: Ensemble   │
                                                                                                              └────────────────────┘
                                                                                          ```

                                                                                          ---

                                                                                          ## 📁 Project Structure

                                                                                          ```
                                                                                          Automated-Road-Damage-Detection/
                                                                                          ├── Notebooks/                    # Jupyter notebooks for analysis and training
                                                                                          │   ├── EDA-RDD.ipynb            # Exploratory Data Analysis
                                                                                          │   ├── baseline-rdd2022.ipynb   # Baseline model implementation
                                                                                          │   ├── Tuning_RDD22.ipynb       # Hyperparameter tuning
                                                                                          │   ├── YOLOv8s_100epochs_Training.ipynb
                                                                                          │   ├── YOLOv8s_1024_Training.ipynb
                                                                                          │   └── Model Evaluation.ipynb    # Comprehensive model evaluation
                                                                                          │
                                                                                          ├── src/                          # Source code
                                                                                          │   ├── inference.py             # Main inference pipeline
                                                                                          │   ├── ensemble.py              # Ensemble logic
                                                                                          │   ├── wbf.py                   # Weighted Boxes Fusion implementation
                                                                                          │   └── __init__.py
                                                                                          │
                                                                                          ├── deployment/                   # Production deployment
                                                                                          │   ├── app.py                   # Flask REST API
                                                                                          │   └── gradio_app.py            # Gradio web interface
                                                                                          │
                                                                                          ├── results/                      # Training and evaluation results
                                                                                          │   ├── yolov8n_640/             # YOLOv8n model results
                                                                                          │   ├── yolov8s_640/             # YOLOv8s @ 640 results
                                                                                          │   ├── yolov8s_1024/            # YOLOv8s @ 1024 results
                                                                                          │   ├── ensemble/                # Ensemble predictions
                                                                                          │   ├── figures/                 # Visualizations for documentation
                                                                                          │   └── metrics/                 # Detailed evaluation metrics
                                                                                          │
                                                                                          ├── models/                       # Trained model weights
                                                                                          │   ├── yolov8n_640.pt
                                                                                          │   ├── yolov8s_640.pt
                                                                                          │   └── yolov8s_1024.pt
                                                                                          │
                                                                                          ├── config/                       # Configuration files
                                                                                          │   └── model_config.yaml        # Model hyperparameters
                                                                                          │
                                                                                          ├── data/                         # Dataset directory
                                                                                          │   └── RDD2022/                 # RDD2022 dataset (download separately)
                                                                                          │       ├── train/
                                                                                          │       │   ├── images/
                                                                                          │       │   └── labels/
                                                                                          │       └── test/
                                                                                          │           ├── images/
                                                                                          │           └── labels/
                                                                                          │
                                                                                          ├── docs/                         # Additional documentation
                                                                                          │   ├── ARCHITECTURE.md          # Detailed architecture explanation
                                                                                          │   ├── TRAINING.md              # Training guide
                                                                                          │   └── API_REFERENCE.md         # API documentation
                                                                                          │
                                                                                          ├── scripts/                      # Utility scripts
                                                                                          │   └── [preprocessing & utility scripts]
                                                                                          │
                                                                                          ├── requirements.txt              # Python dependencies
                                                                                          ├── LICENSE                       # MIT License
                                                                                          ├── .gitignore                   # Git ignore rules
                                                                                          └── README.md                     # This file
                                                                                          ```

                                                                                          ---

                                                                                          ## 📚 Usage Guide

                                                                                          ### Training a Model

                                                                                          ```bash
                                                                                          # Modify config/model_config.yaml with desired parameters
                                                                                          # Then train using the notebook or script:
                                                                                          python scripts/train.py --config config/model_config.yaml
                                                                                          ```

                                                                                          ### Evaluating Models

                                                                                          ```python
                                                                                          from src.inference import RoadDamageDetector

                                                                                          detector = RoadDamageDetector()
                                                                                          metrics = detector.evaluate('path/to/test/dataset')
                                                                                          print(metrics)
                                                                                          ```

                                                                                          ### Ensemble Predictions

                                                                                          ```python
                                                                                          from src.ensemble import EnsemblePredictor

                                                                                          ensemble = EnsemblePredictor(
                                                                                              models=['yolov8n_640.pt', 'yolov8s_640.pt', 'yolov8s_1024.pt'],
                                                                                              weights=[1.0, 1.5, 2.0]
                                                                                          )

                                                                                          predictions = ensemble.predict('image.jpg')
                                                                                          ```

                                                                                          ---

                                                                                          ## 🔧 Configuration

                                                                                          ### Model Configuration (config/model_config.yaml)

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

                                                                                          ## 📥 Dataset Setup

                                                                                          This project uses the **RDD2022 (Road Damage Detection 2022) dataset**.

                                                                                          ### Download Instructions

                                                                                          1. Download from the official source:
                                                                                          2.    - [RDD2022 Repository](https://github.com/sekilab/RoadDamageDetector)
                                                                                            
                                                                                                - 2. Extract and organize into the project:
                                                                                                 
                                                                                                  3. ```bash
                                                                                                     # Create directory structure
                                                                                                     mkdir -p data/RDD2022/{train,test}/{images,labels}

                                                                                                     # Copy dataset
                                                                                                     cp -r /path/to/RDD2022/train/images data/RDD2022/train/
                                                                                                     cp -r /path/to/RDD2022/train/labels data/RDD2022/train/
                                                                                                     cp -r /path/to/RDD2022/test/images data/RDD2022/test/
                                                                                                     cp -r /path/to/RDD2022/test/labels data/RDD2022/test/
                                                                                                     ```
                                                                                                     
                                                                                                     ### Expected Structure
                                                                                                     
                                                                                                     ```
                                                                                                     data/RDD2022/
                                                                                                     ├── train/
                                                                                                     │   ├── images/     # Training images
                                                                                                     │   └── labels/     # Training YOLO format labels (.txt)
                                                                                                     └── test/
                                                                                                         ├── images/     # Test images
                                                                                                         └── labels/     # Test YOLO format labels (.txt)
                                                                                                     ```
                                                                                                     
                                                                                                     ---
                                                                                                     
                                                                                                     ## 🐳 Docker Deployment
                                                                                                     
                                                                                                     ```dockerfile
                                                                                                     # Dockerfile for containerized deployment
                                                                                                     FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

                                                                                                     WORKDIR /app
                                                                                                     COPY requirements.txt .
                                                                                                     RUN pip install -r requirements.txt

                                                                                                     COPY . .

                                                                                                     EXPOSE 5000
                                                                                                     CMD ["python", "deployment/app.py"]
                                                                                                     ```
                                                                                                     
                                                                                                     **Build and Run**:
                                                                                                     ```bash
                                                                                                     docker build -t road-damage-detector .
                                                                                                     docker run -p 5000:5000 road-damage-detector
                                                                                                     ```
                                                                                                     
                                                                                                     ---
                                                                                                     
                                                                                                     ## 📈 Training Pipeline
                                                                                                     
                                                                                                     1. **Data Preparation**: EDA and augmentation (see `Notebooks/EDA-RDD.ipynb`)
                                                                                                     2. 2. **Model Training**: Train individual models at different resolutions
                                                                                                        3. 3. **Hyperparameter Tuning**: Optimize using the tuning notebook
                                                                                                           4. 4. **Evaluation**: Comprehensive metrics and visualizations
                                                                                                              5. 5. **Ensemble Creation**: Combine models with weighted fusion
                                                                                                                 6. 6. **Deployment**: Export and deploy for production use
                                                                                                                   
                                                                                                                    7. ---
                                                                                                                   
                                                                                                                    8. ## 🔬 Experimental Results
                                                                                                                   
                                                                                                                    9. ### Confusion Matrix Analysis
                                                                                                                    10. - Clear separation between damage types
                                                                                                                        - - Low false positive rates
                                                                                                                          - - Robust detection across all classes
                                                                                                                           
                                                                                                                            - ### Per-Class Performance
                                                                                                                            - See `results/metrics/per_class_comparison.csv` for detailed breakdown
                                                                                                                           
                                                                                                                            - ### Visualization
                                                                                                                            - - Training curves: `results/figures/training_curves.png`
                                                                                                                              - - Precision-Recall curves: `results/figures/pr_curve.png`
                                                                                                                                - - Performance comparison: `results/figures/performance_comparison.png`
                                                                                                                                 
                                                                                                                                  - ---
                                                                                                                                  
                                                                                                                                  ## 💡 Key Features
                                                                                                                                  
                                                                                                                                  ✅ **State-of-the-Art Performance**: 66.18% mAP@50 on RDD2022
                                                                                                                                  ✅ **Multiple Deployment Options**: API, UI, and programmatic access
                                                                                                                                  ✅ **Production-Ready**: Error handling, validation, and monitoring
                                                                                                                                  ✅ **Comprehensive Documentation**: From training to deployment
                                                                                                                                  ✅ **Modular Architecture**: Easy to extend and customize
                                                                                                                                  ✅ **GPU Acceleration**: CUDA support for faster inference
                                                                                                                                  
                                                                                                                                  ---
                                                                                                                                  
                                                                                                                                  ## 🤝 Contributing
                                                                                                                                  
                                                                                                                                  Contributions are welcome! Please follow these steps:
                                                                                                                                  
                                                                                                                                  1. Fork the repository
                                                                                                                                  2. 2. Create a feature branch: `git checkout -b feature/improvement`
                                                                                                                                     3. 3. Make your changes and commit: `git commit -am 'Add improvement'`
                                                                                                                                        4. 4. Push to the branch: `git push origin feature/improvement`
                                                                                                                                           5. 5. Submit a pull request
                                                                                                                                             
                                                                                                                                              6. ---
                                                                                                                                             
                                                                                                                                              7. ## 📄 License
                                                                                                                                             
                                                                                                                                              8. This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
                                                                                                                                             
                                                                                                                                              9. ---
                                                                                                                                             
                                                                                                                                              10. ## 📚 Citation
                                                                                                                                             
                                                                                                                                              11. If you use this project in your research or work, please cite:
                                                                                                                                             
                                                                                                                                              12. ```bibtex
                                                                                                                                                  @software{road_damage_detection_2025,
                                                                                                                                                    title = {Automated Road Damage Detection using Multi-Resolution YOLOv8 Ensemble},
                                                                                                                                                    author = {Cyril, Chaitanya Pudota},
                                                                                                                                                    year = {2025},
                                                                                                                                                    url = {https://github.com/Cyril-36/Automated-Road-Damage-Detection},
                                                                                                                                                    note = {GitHub Repository}
                                                                                                                                                  }
                                                                                                                                                  ```
                                                                                                                                                  
                                                                                                                                                  ---
                                                                                                                                                  
                                                                                                                                                  ## 🔗 Resources
                                                                                                                                                  
                                                                                                                                                  - **YOLOv8 Documentation**: [Ultralytics YOLOv8](https://docs.ultralytics.com/)
                                                                                                                                                  - - **RDD2022 Dataset**: [Road Damage Detection 2022](https://github.com/sekilab/RoadDamageDetector)
                                                                                                                                                    - - **Weighted Boxes Fusion**: [WBF Paper](https://arxiv.org/abs/2011.08461)
                                                                                                                                                      - - **PyTorch**: [PyTorch Documentation](https://pytorch.org/docs/)
                                                                                                                                                       
                                                                                                                                                        - ---
                                                                                                                                                        
                                                                                                                                                        ## ❓ Troubleshooting
                                                                                                                                                        
                                                                                                                                                        **Q: Low inference performance on CPU?**
                                                                                                                                                        A: The ensemble inference is GPU-optimized. For CPU inference, use individual smaller models like YOLOv8n.
                                                                                                                                                        
                                                                                                                                                        **Q: Out of memory errors during training?**
                                                                                                                                                        A: Reduce batch size in `config/model_config.yaml` or use gradient accumulation.
                                                                                                                                                        
                                                                                                                                                        **Q: How to use only one model instead of the ensemble?**
                                                                                                                                                        A: Modify `src/inference.py` to load a single model. Inference time will be 3x faster.
                                                                                                                                                        
                                                                                                                                                        ---
                                                                                                                                                        
                                                                                                                                                        ## 📞 Support & Contact
                                                                                                                                                        
                                                                                                                                                        - **Issues**: Open an issue on [GitHub Issues](https://github.com/Cyril-36/Automated-Road-Damage-Detection/issues)
                                                                                                                                                        - - **Email**: cyrilchaitanya@gmail.com
                                                                                                                                                          - - **Documentation**: Check the [docs/](docs/) folder for detailed guides
                                                                                                                                                           
                                                                                                                                                            - ---
                                                                                                                                                            
                                                                                                                                                            ## 🎯 Roadmap
                                                                                                                                                            
                                                                                                                                                            - [ ] Add Real-time video stream processing
                                                                                                                                                            - [ ] - [ ] Implement MLOps pipeline (DVC, CML)
                                                                                                                                                            - [ ] - [ ] Add model quantization for edge deployment
                                                                                                                                                            - [ ] - [ ] Integrate with cloud platforms (AWS, GCP, Azure)
                                                                                                                                                            - [ ] - [ ] Create mobile app for road assessment
                                                                                                                                                            - [ ] - [ ] Add damage severity classification
                                                                                                                                                           
                                                                                                                                                            - [ ] ---
                                                                                                                                                           
                                                                                                                                                            - [ ] ## ⭐ Acknowledgments
                                                                                                                                                           
                                                                                                                                                            - [ ] - **Dataset**: Thanks to the RDD2022 dataset creators
                                                                                                                                                            - [ ] - **YOLOv8**: Ultralytics team for the excellent object detection framework
                                                                                                                                                            - [ ] - **Contributors**: All community members who contribute to this project
                                                                                                                                                           
                                                                                                                                                            - [ ] ---
                                                                                                                                                           
                                                                                                                                                            - [ ] <div align="center">
                                                                                                                                                            
                                                                                                                                                            **Made with ❤️ by Cyril Chaitanya Pudota**
                                                                                                                                                            
                                                                                                                                                            ⭐ If this project helps you, please consider giving it a star!
                                                                                                                                                            
                                                                                                                                                            [Back to Top](#-automated-road-damage-detection)
                                                                                                                                                            
                                                                                                                                                            </div>
