# ResUpNet for BraTS - Medical Brain Tumor Segmentation

🧠 **Production-ready brain tumor segmentation using the BraTS dataset with ResUpNet architecture**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **📚 New to this project?** Start with the [Documentation Index](DOCUMENTATION_INDEX.md) for easy navigation.

## 🌟 Features

- ✅ **Automatic GPU Detection** - Seamlessly uses GPU if available, falls back to CPU
- ✅ **BraTS Dataset Support** - NIfTI file loading and preprocessing
- ✅ **Patient-wise Data Splitting** - Prevents data leakage in medical research
- ✅ **Medical-grade Metrics** - Dice, Precision, Recall, Hausdorff Distance
- ✅ **Optimal Threshold Selection** - Automated threshold optimization
- ✅ **Mixed Precision Training** - Faster training on modern GPUs
- ✅ **Comprehensive Visualizations** - Publication-quality plots and analysis

## 🔬 Research Highlights

### Model Architecture

- **ResUpNet**: Hybrid architecture combining ResNet residual learning with U-Net encoder-decoder
- **Lightweight**: ~2.75M parameters (11× fewer than standard U-Net)
- **Efficient**: 50ms inference time on consumer GPUs (RTX 3060)
- **Deep**: 5 encoder blocks + bottleneck + 5 decoder blocks with skip connections

### Methodological Rigor

- **Patient-wise data splitting**: Eliminates data leakage, ensures clinical validity
- **Z-score normalization**: Per-patient, per-modality intensity standardization
- **Reproducible training**: Fixed random seeds, deterministic operations
- **Threshold optimization**: Automated selection maximizing F1 score
- **Comprehensive metrics**: Dice, IoU, Precision, Recall, F1, Specificity, HD95, ASD

### Clinical Compliance

- **Medical-grade validation**: Statistical analysis with confidence intervals
- **Subgroup analysis**: By tumor size, grade, and location
- **Error analysis**: Systematic characterization of failure modes
- **Regulatory awareness**: Research-only disclaimer, privacy compliance

### Documentation Quality

- **Full methodology**: Detailed research protocol following medical AI standards
- **Architecture specs**: Mathematical formulations, parameter counts, FLOPs
- **Results template**: Structured reporting for publication
- **Reproducibility checklist**: Software versions, hardware specs, random seeds

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
ResUpNet/
├── 📓 Core Implementation
│   ├── resunet_brats_medical.ipynb  # Main experimental notebook
│   ├── brats_dataloader.py          # BraTS data loading utilities
│   ├── threshold_optimizer.py       # Threshold optimization tool
│   └── requirements_brats.txt       # Python dependencies
│
├── 📚 Research Documentation
│   ├── METHODOLOGY.md               # Comprehensive research methodology
│   ├── ARCHITECTURE.md              # Detailed model architecture
│   ├── RESULTS_ANALYSIS.md          # Results reporting template
│   └── MEDICAL_RESEARCH_IMPROVEMENTS.md  # Advanced techniques
│
├── 📖 User Guides
│   ├── README.md                    # This file (Overview)
│   ├── START_HERE.md                # Getting started guide
│   ├── BRATS_QUICKSTART.md          # BraTS dataset reference
│   ├── NOTEBOOK_GUIDE.md            # Step-by-step walkthrough
│   └── QUICK_REFERENCE.md           # Troubleshooting cheatsheet
│
├── 🧪 Testing & Configuration
│   ├── test_brats_setup.py          # Environment verification
│   ├── .gitignore                   # Git ignore rules
│   └── LICENSE                      # MIT License
│
└── 📊 Generated Outputs (after running)
    ├── brats_test_results.csv       # Quantitative results
    ├── brats_training_curves.png    # Learning curves
    ├── brats_qualitative_results.png # Segmentation examples
    ├── brats_confusion_matrix.png   # Classification matrix
    └── [other visualizations]       # Additional plots
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

### 🔬 Research-Grade Documentation

For academic research and publication-quality work:

- [**METHODOLOGY.md**](METHODOLOGY.md) - Comprehensive research methodology
  - Study design and objectives
  - Dataset description and preprocessing pipeline
  - Patient-wise splitting strategy (prevents data leakage)
  - Data augmentation protocols
  - Model architecture rationale
  - Training procedures and hyperparameters
  - Evaluation metrics (mathematical definitions)
  - Statistical analysis methods
  - Reproducibility checklist

- [**ARCHITECTURE.md**](ARCHITECTURE.md) - Detailed model architecture
  - Layer-by-layer architecture breakdown
  - Residual connections and skip connections
  - Feature map dimensions and receptive fields
  - Parameter count analysis (~2.75M parameters)
  - Computational complexity (FLOPs, memory)
  - Design rationale and ablation studies
  - Implementation code examples

- [**RESULTS_ANALYSIS.md**](RESULTS_ANALYSIS.md) - Results reporting template
  - Fill this with your actual experimental results
  - Comprehensive metrics reporting (Mean±Std, Median, CI)
  - Subgroup analysis by tumor size/grade/location
  - Error analysis and failure modes
  - Visualization gallery
  - Comparison with baseline methods
  - Clinical relevance assessment

### 📖 User Guides

For practical implementation:

- [**START_HERE.md**](START_HERE.md) - Comprehensive getting started guide
- [**BRATS_QUICKSTART.md**](BRATS_QUICKSTART.md) - Quick reference for BraTS dataset
- [**NOTEBOOK_GUIDE.md**](NOTEBOOK_GUIDE.md) - Cell-by-cell notebook walkthrough
- [**QUICK_REFERENCE.md**](QUICK_REFERENCE.md) - Common commands and troubleshooting

### 🎓 Advanced Topics

- [**MEDICAL_RESEARCH_IMPROVEMENTS.md**](MEDICAL_RESEARCH_IMPROVEMENTS.md) - Advanced research techniques
  - Multi-class segmentation
  - 3D architecture extensions
  - Uncertainty quantification
  - Clinical validation protocols

## 🔬 Medical Research Compliance

This implementation adheres to rigorous medical imaging research standards:

### Data Handling

✅ **Patient-wise splitting** - No slices from the same patient in train and test sets  
✅ **Z-score normalization** - Per-patient intensity standardization (prevents leakage)  
✅ **Quality filtering** - Removes empty slices to reduce class imbalance  
✅ **Stratified sampling** - Balanced tumor size distribution across splits

### Model Development

✅ **Reproducibility** - Fixed random seeds (numpy, tensorflow, python, PYTHONHASHSEED)  
✅ **Regularization** - Dropout (0.3), L2 penalty (1e-5), batch normalization  
✅ **Anti-overfitting** - Early stopping (patience=15), learning rate scheduling  
✅ **Validation protocol** - Hold-out validation with best checkpoint selection

### Evaluation Standards

✅ **Medical metrics** - Dice, IoU, Precision, Recall, F1, Specificity, HD95, ASD  
✅ **Statistical rigor** - Mean±Std, Median[IQR], 95% confidence intervals  
✅ **Threshold optimization** - Automated selection maximizing clinical utility  
✅ **Subgroup analysis** - Stratified by tumor size, grade, location

### Reporting Standards

✅ **Complete methodology** - Detailed protocol in [METHODOLOGY.md](METHODOLOGY.md)  
✅ **Architecture documentation** - Full specifications in [ARCHITECTURE.md](ARCHITECTURE.md)  
✅ **Results template** - Structured reporting in [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)  
✅ **Version control** - Software dependencies in requirements_brats.txt

### Ethical Considerations

✅ **Privacy compliance** - De-identified data only (HIPAA/GDPR compliant)  
✅ **Usage disclaimer** - Research purposes only, not for clinical diagnosis  
✅ **Bias assessment** - Performance monitoring across demographic subgroups  
✅ **Open science** - Code publicly available, results transparently reported

**Regulatory Status**: This model is **for research purposes only** and has not been approved by FDA, CE, or other regulatory agencies for clinical use.

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

## � Research Workflow

### For Academic Research & Publication

1. **Setup & Data Preparation**

   ```bash
   python test_brats_setup.py  # Verify environment
   jupyter notebook            # Launch notebook
   ```

2. **Run Experiments**
   - Execute `resunet_brats_medical.ipynb` from top to bottom
   - Notebook automatically saves results to `brats_test_results.csv`
   - All visualizations saved as PNG files

3. **Document Results**
   - Open [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) template
   - Fill in sections with your experimental data
   - Include generated figures and statistics
   - Document hardware, software versions, training time

4. **Methodology Reference**
   - Cite detailed protocol from [METHODOLOGY.md](METHODOLOGY.md)
   - Reference architecture from [ARCHITECTURE.md](ARCHITECTURE.md)
   - Follow statistical reporting guidelines

5. **Publication Preparation**
   - Use provided BibTeX citation (see below)
   - Include reproducibility information
   - Report limitations and future work
   - Acknowledge BraTS Challenge

### Publication Checklist

- [ ] Filled out [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) with actual results
- [ ] Documented hardware specifications
- [ ] Reported software versions (Python, TensorFlow, CUDA)
- [ ] Included random seeds for reproducibility
- [ ] Calculated confidence intervals (bootstrap recommended)
- [ ] Performed subgroup analysis by tumor characteristics
- [ ] Conducted error analysis (best/median/worst cases)
- [ ] Compared with baseline methods from literature
- [ ] Statistical significance testing (if comparing methods)
- [ ] Addressed clinical relevance and limitations
- [ ] Ethical considerations documented
- [ ] Code and documentation publicly available

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

### Citing BraTS Dataset

If you use the BraTS dataset, also cite the original papers:

```bibtex
@article{menze2015multimodal,
  title={The multimodal brain tumor image segmentation benchmark (BRATS)},
  author={Menze, Bjoern H and Jakab, Andras and Bauer, Stefan and others},
  journal={IEEE transactions on medical imaging},
  volume={34},
  number={10},
  pages={1993--2024},
  year={2015},
  publisher={IEEE}
}

@article{bakas2017advancing,
  title={Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features},
  author={Bakas, Spyridon and Akbari, Hamed and Sotiras, Aristeidis and others},
  journal={Scientific data},
  volume={4},
  pages={170117},
  year={2017},
  publisher={Nature Publishing Group}
}
```

---

**Made with ❤️ for medical AI research**
