# ResUpNet: Advanced Residual Attention U-Net for Brain MRI Tumor Segmentation

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

## 📋 Overview

**ResUpNet** is a novel deep learning architecture combining the strengths of ResNet-50 encoder with attention-gated U-Net decoder for precise brain tumor segmentation in MRI scans. This architecture introduces several innovative components specifically designed for medical image segmentation tasks, particularly targeting Low-Grade Glioma (LGG) detection.

### Key Innovations

- **Hybrid Encoder-Decoder Architecture**: Leverages pre-trained ResNet-50 as a powerful feature extractor while maintaining full spatial resolution recovery
- **Attention Gate Mechanism**: Implements spatial attention gates at each decoder level to suppress irrelevant feature activations and highlight salient features
- **Residual Decoder Blocks**: Employs residual connections in the decoder path to facilitate gradient flow and improve feature representation
- **Hybrid Loss Function**: Combines Dice Loss and Focal Loss for handling class imbalance and improving boundary precision
- **Multi-Resolution Skip Connections**: Utilizes multi-scale features from different ResNet stages with attention refinement

## 🎯 Applications

This model is specifically designed for:

1. **Clinical Brain Tumor Segmentation**: Automated detection and delineation of Low-Grade Gliomas (LGG) in MRI scans
2. **Medical Image Analysis Research**: Baseline architecture for comparative studies in semantic segmentation
3. **Computer-Aided Diagnosis (CAD)**: Integration into clinical decision support systems
4. **Educational Purposes**: Demonstration of advanced deep learning techniques in medical imaging
5. **Radiomics Studies**: Automated tumor region extraction for downstream feature analysis

## 🏗️ Architecture Details

### Model Components

```
Input (256×256×1) → RGB Replication (256×256×3)
                    ↓
              ResNet-50 Encoder
                    ↓
    [conv1_relu, conv2_block3, conv3_block4, conv4_block6, conv5_block3]
                    ↓
            Attention-Gated Decoder
                    ↓
          Residual Conv Blocks (512→256→128→64→32)
                    ↓
         Output: Binary Mask (256×256×1)
```

### Technical Specifications

- **Encoder**: ResNet-50 (pre-trained on ImageNet) - 23.5M parameters
- **Decoder**: Custom attention-gated upsampling path - 4.2M parameters
- **Total Parameters**: ~27.7M (trainable/frozen configurable)
- **Input Resolution**: 256×256 pixels (grayscale MRI)
- **Output**: Binary segmentation mask (tumor/non-tumor)
- **Loss Function**: Combo Loss (Dice + Binary Cross-Entropy) / Hybrid Loss (Dice + Focal)
- **Optimizer**: Adam (learning rate: 1e-4)

### Performance Metrics

The model is evaluated using:

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard Index for segmentation accuracy
- **Precision & Recall**: Pixel-wise classification performance
- **F1-Score**: Harmonic mean of precision and recall

## 📊 Dataset

The model is trained on the **Brain MRI Segmentation Dataset** (Kaggle 3M):

- **Source**: TCGA Lower Grade Glioma Collection
- **Total Scans**: 3,929 brain MRI slices from 110 patients
- **Format**: TIFF images with corresponding binary masks
- **Resolution**: Variable (resized to 256×256)
- **Split Ratio**: 70% Training / 15% Validation / 15% Testing

### Data Attribution

- **Original Dataset**: [Brain MRI Segmentation - Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Data Provider**: Mateusz Buda et al.
- **Original Paper**: [Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm](https://doi.org/10.1016/j.compbiomed.2019.05.002)
- **TCGA Data Portal**: [The Cancer Genome Atlas](https://www.cancer.gov/tcga)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM for data processing

### GPU Training Notes (Windows / WSL2 / Colab)

- **Google Colab (easiest):** set `Runtime > Change runtime type > GPU`, then run the notebook.
- **Windows (native):** recent TensorFlow releases do **not** support NVIDIA GPU acceleration on native Windows Python.
    Use **WSL2 (recommended)** or a Linux machine for TensorFlow GPU training.
- **WSL2 / Linux:** install TensorFlow with CUDA dependencies via:
    - `pip install "tensorflow[and-cuda]>=2.15,<2.16"`

To verify TensorFlow sees your GPU:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

### WSL2 GPU Setup (Recommended for Windows)

This is the most reliable way to train this TensorFlow model on an NVIDIA GPU from a Windows machine.

1. **Update Windows + enable WSL2**
    - Install WSL2 and an Ubuntu distro.
    - In PowerShell (admin):
      - `wsl --install`
      - `wsl --set-default-version 2`

2. **Install the NVIDIA Windows driver with WSL support**
    - Install the latest NVIDIA driver for your GPU from NVIDIA.
    - Reboot.

3. **Verify GPU is visible inside WSL2**
    - Open Ubuntu (WSL) and run:
      - `nvidia-smi`
    - If this fails, fix the driver/WSL setup before continuing.

4. **Create a Python environment in WSL2 and install deps**
    - From your repo folder inside WSL2 (recommended: clone the repo in WSL2, or open it via `/mnt/c/...`):
      - `python3 -m venv .venv`
      - `source .venv/bin/activate`
      - `pip install -U pip`
      - `pip install -r requirements.txt`
      - `pip install "tensorflow[and-cuda]>=2.15,<2.16"`

5. **Run the notebook in WSL2**
    - Option A (VS Code recommended): install the **Remote - WSL** extension, then:
      - `code .` from inside WSL2
      - Select the WSL Python interpreter (`.venv`)
      - Run `lggsegment_cpu.ipynb`
    - Option B (browser Jupyter):
      - `pip install jupyter`
      - `jupyter notebook`

6. **Confirm TensorFlow sees the GPU**
    - In the notebook, the TensorFlow GPU cell should print a non-empty GPU list.

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/resunet.git
   cd resunet
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Load Preprocessed Data (Recommended)

```python
import numpy as np

# Load preprocessed splits
X_train = np.load('processed_splits/X_train.npy')
y_train = np.load('processed_splits/y_train.npy')
X_val = np.load('processed_splits/X_val.npy')
y_val = np.load('processed_splits/y_val.npy')
X_test = np.load('processed_splits/X_test.npy')
y_test = np.load('processed_splits/y_test.npy')
```

#### Option 2: Full Training Pipeline

Open and run `lggsegment_cpu.ipynb` in Jupyter Notebook or Google Colab:

```bash
jupyter notebook lggsegment_cpu.ipynb
```

The notebook includes:

1. Data loading and preprocessing
2. Model architecture definition
3. Training with callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
4. Evaluation and visualization
5. Prediction on test samples

### Model Training

```python
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Define callbacks
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_dice_coef', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
early_stop = EarlyStopping(monitor='val_dice_coef', patience=15, mode='max', restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint, reduce_lr, early_stop]
)
```

## 📦 Project Structure

```
resunet/
├── lggsegment.ipynb          # Main training notebook
├── requirements.txt           # Python dependencies
├── processed_splits/          # Preprocessed data splits
│   ├── X_train.npy           # Training images
│   ├── y_train.npy           # Training masks
│   ├── X_val.npy             # Validation images
│   ├── y_val.npy             # Validation masks
│   ├── X_test.npy            # Test images
│   └── y_test.npy            # Test masks
├── venv/                      # Virtual environment (not tracked)
├── .gitignore                 # Git ignore file
├── LICENSE                    # License file
└── README.md                  # This file
```

## 🔬 Technical Details

### Attention Gate Mechanism

The attention gates are implemented to focus on relevant spatial locations:

```python
def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, 1)(x)
    phi_g = Conv2D(inter_channels, 1)(g)
    add = Add()([theta_x, phi_g])
    relu = Activation('relu')(add)
    psi = Conv2D(1, 1)(relu)
    sig = Activation('sigmoid')(psi)
    return Multiply()([x, sig])
```

### Hybrid Loss Function

Combines Dice Loss and Focal Loss for robust training:

```python
def hybrid_loss(alpha=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        return alpha * dice_loss(y_true, y_pred) + (1-alpha) * focal_loss(y_true, y_pred)
    return loss
```

## 📈 Expected Results

Typical performance metrics on test set:

- **Dice Coefficient**: 0.85-0.92
- **IoU Score**: 0.75-0.85
- **Precision**: 0.88-0.94
- **Recall**: 0.82-0.90
- **F1-Score**: 0.85-0.92

_Note: Results may vary based on training configuration, data splits, and hardware._

## 🙏 Acknowledgments

This project builds upon and acknowledges the following resources:

### Foundational Work

- **U-Net Architecture**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- **Attention U-Net**: Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL 2018.

### Dataset & Research

- **Brain MRI Dataset**: Buda, M., Saha, A., & Mazurowski, M.A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. Computers in Biology and Medicine.
- **TCGA Research Network**: The Cancer Genome Atlas Program - NCI

### Deep Learning Frameworks

- **TensorFlow/Keras**: Google Brain Team & Contributors
- **PyTorch**: Meta AI Research (for compatibility extensions)
- **OpenCV**: Open Source Computer Vision Library

### Community Contributions

- Kaggle community for dataset curation and benchmarks
- Medical imaging research community for validation methodologies

## 📝 Citation

If you use this work in academic research, please cite:

```bibtex
@software{resunet2026,
  author = {Jagan Kumar Hotta},
  title = {ResUpNet: Advanced Residual Attention U-Net for Brain MRI Tumor Segmentation},
  year = {2026},
  url = {https://github.com/techySphinx/ResUpNet},
  note = {License required for commercial and research use}
}
```

## ⚖️ License

This project is released under a **Proprietary Commercial License**. See the [LICENSE](LICENSE) file for full terms.

**Summary**:

- ✅ Personal learning and educational use permitted
- ❌ Commercial use requires paid license
- ❌ Research publication requires prior authorization
- ❌ Patent claims require licensing agreement
- ❌ Redistribution without permission prohibited

For licensing inquiries, please contact: jaganhotta357@outlook.com

## 📞 Contact & Support

- **Author**: Jagan Kumar Hotta
- **Email**: jaganhotta357@outlook.com
- **GitHub**: [@techySphinx](https://github.com/techySphinx)
- **Issues**: [GitHub Issues](https://github.com/techySphinx/resunet/issues)

## 🔮 Future Enhancements

- [ ] Multi-class segmentation support (tumor sub-regions)
- [ ] 3D volumetric segmentation
- [ ] Model quantization for edge deployment
- [ ] ONNX export for cross-platform inference
- [ ] Web-based inference API
- [ ] Integration with medical DICOM viewers

---

**Disclaimer**: This software is provided for research and educational purposes. It is not intended for clinical diagnosis or medical decision-making without proper validation and regulatory approval.

**Version**: 1.0.0 | **Last Updated**: January 2026
