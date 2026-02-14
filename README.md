# ResUpNet for BraTS - Medical Brain Tumor Segmentation

🧠 **Production-ready brain tumor segmentation using the BraTS dataset with ResUpNet architecture**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

- ✅ **Automatic GPU Detection** - Seamlessly uses GPU if available, falls back to CPU
- ✅ **BraTS Dataset Support** - NIfTI file loading and preprocessing
- ✅ **Patient-wise Data Splitting** - Prevents data leakage in medical research
- ✅ **Medical-grade Metrics** - Dice, Precision, Recall, Hausdorff Distance
- ✅ **Optimal Threshold Selection** - Automated threshold optimization
- ✅ **Mixed Precision Training** - Faster training on modern GPUs
- ✅ **Comprehensive Visualizations** - Publication-quality plots and analysis

## 📊 Expected Results

| Metric           | Score Range |
| ---------------- | ----------- |
| Dice Coefficient | 0.88 - 0.92 |
| Precision        | 0.86 - 0.92 |
| Recall           | 0.85 - 0.90 |
| F1 Score         | 0.86 - 0.91 |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/techySPHINX/ResUpNet.git
cd ResUpNet
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements_brats.txt
```

### 3. Download BraTS Dataset

**Option A: Kaggle (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Download BraTS 2020 dataset
kaggle datasets download -d awsaf49/brats2020-training-data
unzip brats2020-training-data.zip
```

**Option B: Official BraTS Challenge**

- Visit: https://www.med.upenn.edu/cbica/brats2020/data.html
- Register and download the training dataset
- Extract to a folder (e.g., `BraTS2021_Training_Data`)

### 4. Run the Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open resunet_brats_medical.ipynb
# Run cells from top to bottom
```

## 📁 Project Structure

```
resunet/
├── resunet_brats_medical.ipynb  # Main notebook (START HERE)
├── brats_dataloader.py           # BraTS data loading utilities
├── threshold_optimizer.py        # Threshold optimization tool
├── requirements_brats.txt        # Python dependencies
├── test_brats_setup.py          # Environment test script
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── README.md                    # This file
├── START_HERE.md               # Detailed getting started guide
├── BRATS_QUICKSTART.md         # Quick reference for BraTS
├── NOTEBOOK_GUIDE.md           # Step-by-step notebook guide
├── QUICK_REFERENCE.md          # Cheat sheet for common tasks
└── MEDICAL_RESEARCH_IMPROVEMENTS.md  # Advanced research tips
```

## 🎯 Workflow Overview

### Step 1: Environment Setup

The notebook automatically detects and configures:

- ✅ GPU/CPU availability
- ✅ TensorFlow device configuration
- ✅ Mixed precision training (if GPU available)
- ✅ Memory growth settings

### Step 2: Data Loading

Two options available:

- **Option A**: Load preprocessed data (fast, if already processed)
- **Option B**: Process raw BraTS dataset (first time, ~1-2 hours)

### Step 3: Data Preprocessing

- Patient-wise z-score normalization
- Patient-wise train/val/test splitting (prevents leakage)
- Medical image augmentation
- Quality filtering (removes empty slices)

### Step 4: Model Training

- ResUpNet architecture with skip connections
- Binary segmentation (tumor vs. background)
- Dice loss with focal component
- Learning rate scheduling
- Model checkpointing

### Step 5: Threshold Optimization

- Automated optimal threshold selection
- Balances precision and recall
- Maximizes F1 score

### Step 6: Evaluation & Visualization

- Comprehensive metrics calculation
- Hausdorff Distance (HD95)
- Statistical analysis
- Publication-quality plots

## 🔧 Configuration

### GPU Configuration

The notebook automatically detects GPU. No manual configuration needed!

```python
# Automatic GPU detection in notebook cell 2
# Will use GPU if available, otherwise CPU
# Mixed precision automatically enabled for modern GPUs
```

### Dataset Path Configuration

Update the dataset path in the notebook:

```python
# For local machine
BRATS_ROOT = "C:/Users/KIIT/Desktop/Datasets/BraTS2021_Training_Data"

# For Google Colab
BRATS_ROOT = "/content/drive/MyDrive/Datasets/BraTS2021_Training_Data"
```

### Training Hyperparameters

```python
BATCH_SIZE = 16          # Increase if you have more GPU memory
EPOCHS = 50              # Adjust based on convergence
LEARNING_RATE = 1e-4     # Adam optimizer learning rate
IMG_SIZE = (256, 256)    # Input image dimensions
```

## 📋 Requirements

### Hardware

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU (8GB+ VRAM)
- **Optimal**: 32GB RAM, NVIDIA RTX 3080/4080 (12GB+ VRAM)

### Software

- Python 3.8+
- TensorFlow 2.13+ (with GPU support)
- CUDA 11.8+ and cuDNN 8.6+ (for GPU)
- Jupyter Notebook

## 🧪 Testing Your Setup

Run the setup test script:

```bash
python test_brats_setup.py
```

This will verify:

- ✅ Python version
- ✅ TensorFlow installation
- ✅ GPU availability
- ✅ Required packages
- ✅ CUDA/cuDNN (if GPU)

## 📚 Documentation

- [**START_HERE.md**](START_HERE.md) - Comprehensive getting started guide
- [**BRATS_QUICKSTART.md**](BRATS_QUICKSTART.md) - Quick reference for BraTS dataset
- [**NOTEBOOK_GUIDE.md**](NOTEBOOK_GUIDE.md) - Cell-by-cell notebook walkthrough
- [**QUICK_REFERENCE.md**](QUICK_REFERENCE.md) - Common commands and troubleshooting
- [**MEDICAL_RESEARCH_IMPROVEMENTS.md**](MEDICAL_RESEARCH_IMPROVEMENTS.md) - Research tips

## 🔬 Medical Research Compliance

This implementation follows medical imaging best practices:

✅ **Patient-wise splitting** - Prevents data leakage  
✅ **Z-score normalization** - Per-patient intensity standardization  
✅ **Medical metrics** - Dice, HD95, ASD, Precision, Recall  
✅ **Threshold optimization** - Maximizes clinical utility  
✅ **Reproducibility** - Fixed random seeds, version pinning

## 💡 Common Issues & Solutions

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Verify TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Update CUDA/cuDNN if needed
# Visit: https://www.tensorflow.org/install/gpu
```

### Out of Memory Error

```python
# Reduce batch size in notebook
BATCH_SIZE = 8  # or 4

# Enable memory growth (already automatic in notebook)
```

### Dataset Not Found

```python
# Verify dataset path
import os
print(os.path.exists(BRATS_ROOT))

# Check directory structure
# Should have folders like: BraTS2021_00001, BraTS2021_00002, etc.
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BraTS Challenge** - Multimodal Brain Tumor Segmentation Challenge
- **Medical Image Computing** - Research community
- **TensorFlow Team** - Deep learning framework

## 📧 Contact

- **Author**: techySPHINX
- **GitHub**: [@techySPHINX](https://github.com/techySPHINX)
- **Repository**: [ResUpNet](https://github.com/techySPHINX/ResUpNet)

## 📈 Citation

If you use this code in your research, please cite:

```bibtex
@software{resunet_brats2024,
  author = {techySPHINX},
  title = {ResUpNet for BraTS: Medical Brain Tumor Segmentation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/techySPHINX/ResUpNet}
}
```

---

**Made with ❤️ for medical AI research**
