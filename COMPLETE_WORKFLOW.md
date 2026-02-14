# 🚀 BraTS ResUpNet - Complete Setup & Execution Guide

## Step-by-Step Workflow: Download to Final Results

### ✅ Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] NVIDIA GPU (optional but recommended)
- [ ] CUDA 11.8+ and cuDNN 8.6+ (for GPU)
- [ ] 20+ GB free disk space (for dataset)
- [ ] Stable internet connection (for dataset download)

---

## 🔧 Step 1: Environment Setup

### 1.1 Clone Repository

```powershell
cd C:\Users\KIIT\Desktop\open-source
git clone https://github.com/techySPHINX/ResUpNet.git
cd ResUpNet
```

### 1.2 Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1.3 Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements_brats.txt

# Verify installation
python test_brats_setup.py
```

**Expected Output:**

```
✅ Python version: 3.x.x
✅ TensorFlow version: 2.x.x
✅ GPU detected: NVIDIA GeForce RTX ...
✅ CUDA available: True
✅ All required packages installed
```

---

## 📥 Step 2: Download BraTS Dataset

### Option A: Kaggle (Recommended - Easier)

#### 2.1 Install Kaggle CLI

```powershell
pip install kaggle
```

#### 2.2 Configure Kaggle API

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save `kaggle.json` to: `C:\Users\KIIT\.kaggle\kaggle.json`

#### 2.3 Download Dataset

```powershell
# Create dataset directory
mkdir C:\Users\KIIT\Desktop\Datasets
cd C:\Users\KIIT\Desktop\Datasets

# Download BraTS 2020 dataset (~7GB)
kaggle datasets download -d awsaf49/brats2020-training-data

# Extract
Expand-Archive brats2020-training-data.zip -DestinationPath .

# Verify structure
ls BraTS2021_Training_Data
```

### Option B: Official BraTS Website (Alternative)

1. Visit: https://www.med.upenn.edu/cbica/brats2020/data.html
2. Register account
3. Download "Training Data" (BraTS2020 or BraTS2021)
4. Extract to: `C:\Users\KIIT\Desktop\Datasets\BraTS2021_Training_Data`

### 2.4 Verify Dataset Structure

```powershell
# Check that you have folders like:
# BraTS2021_00001/
# BraTS2021_00002/
# ...
# Each containing: *_flair.nii.gz, *_seg.nii.gz, etc.
```

---

## 📓 Step 3: Launch Jupyter Notebook

### 3.1 Start Jupyter

```powershell
# Make sure you're in the project directory
cd C:\Users\KIIT\Desktop\open-source\ResUpNet

# Launch Jupyter
jupyter notebook
```

### 3.2 Open Notebook

- Browser will open automatically
- Navigate to: `resunet_brats_medical.ipynb`
- Click to open

---

## 🎯 Step 4: Execute Notebook Cells (One by One)

### Cell 1: Introduction (Markdown)

- Just read the introduction
- No execution needed

### Cell 2: Environment Detection

```python
# Run this cell
# It will automatically detect Colab vs Local
```

**Expected Output:**

```
✅ Running on Local Machine
```

### Cell 3: 🔥 Automatic GPU Configuration

```python
# Run this cell - NO MANUAL CHANGES NEEDED!
# It automatically detects and configures GPU
```

**Expected Output:**

```
🔍 TensorFlow Device Status:
TensorFlow Version: 2.x.x
Platform: Windows
Built with CUDA: True
GPUs detected: 1
✅ GPU detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ✓ Memory growth enabled for /physical_device:GPU:0
✅ Using single GPU strategy
✅ Mixed precision enabled (float16) for faster training
🧪 Running GPU sanity test...
✅ GPU sanity test passed (sum: ...)
🎯 Final Configuration: GPU with OneDeviceStrategy
   Mixed Precision: True
```

### Cell 4: Load/Preprocess Data

**IMPORTANT:** Choose ONE option:

#### Option A: Load Preprocessed (if you've run this before)

```python
# Run cell 5 - Load preprocessed data
```

#### Option B: Process Raw Dataset (First Time - ~1-2 hours)

```python
# Run cell 6 - Process raw BraTS dataset
# ⚠️ This takes 1-2 hours for full dataset!
```

**What happens:**

- Loads NIfTI files from BraTS dataset
- Applies patient-wise z-score normalization
- Filters slices with tumors (min 50 pixels)
- Splits data patient-wise (70/15/15)
- Saves preprocessed data for future use

**Expected Output:**

```
📂 BraTS dataset path: C:/Users/KIIT/Desktop/Datasets/BraTS2021_Training_Data
⏳ Loading and preprocessing BraTS dataset...
Processing patient 1/1251...
...
✅ Preprocessing complete!
Train: (8000, 256, 256, 1) images
Val:   (1700, 256, 256, 1) images
Test:  (1700, 256, 256, 1) images
```

### Cell 7: Visualize Samples

```python
# Run to see sample images and masks
```

### Cell 8: Data Augmentation Setup

```python
# Run to configure augmentation pipeline
```

### Cell 9: Define ResUpNet Architecture

```python
# Run to define model functions
```

### Cell 10: 🏗️ Build & Compile Model

```python
# Run to create and compile model
# Uses GPU automatically if available
```

**Expected Output:**

```
✅ Model compiled successfully
Strategy: OneDeviceStrategy
GPUs: [PhysicalDevice(name='/physical_device:GPU:0'...)]
Model: "ResUpNet_BraTS"
Total params: 24,456,193
Trainable params: 24,410,561
Non-trainable params: 45,632
```

### Cell 11: Define Evaluation Metrics

```python
# Run to define metrics functions
```

### Cell 12: 🏋️ Train Model (~2-4 hours on GPU)

```python
# Run to start training
# Automatic GPU utilization
# Mixed precision for faster training
```

**Expected Progress:**

```
Epoch 1/50
500/500 [==============================] - 180s 360ms/step
  loss: 0.1234 - dice_coef: 0.8423 - val_dice_coef: 0.8156
Epoch 2/50
500/500 [==============================] - 165s 330ms/step
  loss: 0.0987 - dice_coef: 0.8756 - val_dice_coef: 0.8598
...
```

**Training Complete (~40-50 epochs)**

### Cell 13: Plot Training Curves

```python
# Run to visualize training progress
```

**Output:** `brats_training_curves.png`

### Cell 14: 🎯 Find Optimal Threshold

```python
# Run threshold optimization
# CRITICAL for best precision/recall
```

**Expected Output:**

```
🎯 Optimal Threshold: 0.423
   F1 Score: 0.8945
   Dice: 0.8912
   Precision: 0.8876
   Recall: 0.9014
```

### Cell 15-20: Comprehensive Evaluation

```python
# Run all evaluation cells
# Computes test set metrics
# Generates visualizations
```

**Generated Files:**

- `best_resupnet_brats.keras` - Trained model
- `brats_test_results.csv` - Detailed metrics
- `threshold_optimization_analysis.png`
- `brats_metrics_distribution.png`
- `brats_qualitative_results.png`
- `brats_training_curves.png`
- `brats_medical_research_summary.txt`

### Final Cell: Summary & Research Template

```python
# Read the research paper template
# Copy metrics for your publication
```

---

## 📊 Expected Final Results

| Metric           | Target    | Your Result |
| ---------------- | --------- | ----------- |
| Dice Coefficient | 0.88-0.92 | \_\_\_      |
| Precision        | 0.86-0.92 | \_\_\_      |
| Recall           | 0.85-0.90 | \_\_\_      |
| F1 Score         | 0.86-0.91 | \_\_\_      |
| Specificity      | 0.95+     | \_\_\_      |

---

## 🎓 For Quick Testing (Before Full Run)

If you want to test quickly first:

1. In Cell 6 (data loading), uncomment:

```python
images, masks, patient_info = loader.load_dataset(
    max_patients=50,  # ⭐ Uncomment this line
    verbose=True
)
```

2. In Cell 12 (training), reduce epochs:

```python
epochs=10,  # Change from 50 to 10
```

This will:

- Use only 50 patients (~500 slices)
- Train for 10 epochs (~30 minutes on GPU)
- Let you verify everything works before full training

---

## 🐛 Common Issues & Solutions

### Issue 1: GPU Not Detected

```powershell
# Check NVIDIA driver
nvidia-smi

# Reinstall TensorFlow with GPU
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

### Issue 2: Out of Memory (OOM)

In notebook cell, reduce batch size:

```python
BATCH_SIZE = 8  # Reduce from 16
```

### Issue 3: Dataset Path Not Found

Update path in Cell 6:

```python
BRATS_ROOT = "C:/Users/KIIT/Desktop/Datasets/BraTS2021_Training_Data"
# Verify this exact path exists!
```

### Issue 4: Slow Training

Check GPU utilization:

```powershell
# In separate terminal
nvidia-smi -l 1
# Should show ~90%+ GPU utilization
```

---

## 📈 Training Time Estimates

| Hardware    | Preprocessing | Training (50 epochs) | Total     |
| ----------- | ------------- | -------------------- | --------- |
| CPU only    | 3-4 hours     | 20-30 hours          | ~34 hours |
| GTX 1660 Ti | 2-3 hours     | 6-8 hours            | ~11 hours |
| RTX 3060    | 1.5-2 hours   | 3-4 hours            | ~6 hours  |
| RTX 3080    | 1-1.5 hours   | 2-3 hours            | ~4 hours  |
| RTX 4090    | 45-60 min     | 1-2 hours            | ~3 hours  |

---

## ✅ Completion Checklist

After running all cells, you should have:

- [x] Model trained on BraTS dataset
- [x] GPU automatically detected and used
- [x] Optimal threshold found: `___`
- [x] Test Dice score: `___`
- [x] All visualization images saved
- [x] Research summary generated
- [x] Model saved: `best_resupnet_brats.keras`

**🎉 Congratulations! Your model is ready for medical research publication!**

---

## 🔄 Next Steps

1. **Improve Results:**
   - Use full dataset (all patients)
   - Train for more epochs (100+)
   - Ensemble multiple models
   - Try other modalities (T1, T2, T1ce)

2. **Advanced Features:**
   - Multi-class segmentation (whole tumor, tumor core, enhancing tumor)
   - 3D segmentation (volumetric)
   - Post-processing (morphological operations)
   - Uncertainty quantification

3. **Deployment:**
   - Export to ONNX for fast inference
   - Create web app with Streamlit
   - Docker containerization
   - Clinical integration

---

## 📚 Additional Resources

- [START_HERE.md](START_HERE.md) - Detailed setup guide
- [BRATS_QUICKSTART.md](BRATS_QUICKSTART.md) - Dataset info
- [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) - Cell-by-cell guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Commands cheatsheet

---

**Need help?** Check the README.md or open an issue on GitHub!
