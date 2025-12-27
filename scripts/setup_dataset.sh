#!/bin/bash
# Setup RDD2022 dataset

echo "📊 Setting up RDD2022 Dataset..."

# Create data directory
mkdir -p data

# Instructions
echo "Please download the RDD2022 dataset from:"
echo "https://github.com/sekilab/RoadDamageDetector"
echo ""
echo "After downloading, place it in: data/RDD2022/"
echo ""
echo "Expected structure:"
echo "data/RDD2022/"
echo "├── train/"
echo "│   ├── images/"
echo "│   └── labels/"
echo "└── test/"
echo "    ├── images/"
echo "    └── labels/"
