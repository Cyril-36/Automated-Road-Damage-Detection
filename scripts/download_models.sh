#!/bin/bash
# Download pretrained model weights

echo "🔽 Downloading Road Damage Detection Models..."

# Create models directory if it doesn't exist
mkdir -p models

# TODO: Add download links when models are uploaded
echo "⚠️  Please download models manually from:"
echo "   - Google Drive: https://drive.google.com/..."
echo "   - HuggingFace: huggingface.co/Cyril-36/road-damage-detection"
echo ""
echo "Expected files:"
echo "   - models/yolov8n_640.pt"
echo "   - models/yolov8s_640.pt"
echo "   - models/yolov8s_1024.pt"
