# Data Directory

This directory contains datasets and sample images for road damage detection.

## Structure

```
data/
├── sample_images/     # Sample road images (included in repo)
├── RDD2022/          # Full RDD2022 dataset (download separately)
├── train/            # Training data
├── test/             # Test data
└── val/              # Validation data
```

## Sample Images

The `sample_images/` directory contains 5 sample images from different countries:
- United States
- India
- Japan
- Norway
- China

These are included in the repository for testing and demonstration purposes.

## RDD2022 Dataset

The full RDD2022 dataset is **NOT** included in this repository due to size constraints.

### Download Instructions

1. Visit the official RDD2022 repository:
   https://github.com/sekilab/RoadDamageDetector

2. Follow their instructions to download the dataset

3. Place the downloaded data in `data/RDD2022/` with this structure:
   ```
   data/RDD2022/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### Dataset Statistics

- **Total Images**: 38,385
- **Countries**: 7 (Japan, India, USA, Czech Republic, Norway, China, Croatia)
- **Damage Classes**: 5
- **Train/Test Split**: 30,708 / 7,677 (80/20)
- **Annotation Format**: YOLO format

## Citation

If you use the RDD2022 dataset, please cite:

```bibtex
@article{arya2022rdd2022,
  title={RDD2022: A multi-national image dataset for automatic Road Damage Detection},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and others},
  journal={arXiv preprint arXiv:2209.08538},
  year={2022}
}
```
