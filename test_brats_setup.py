"""
Quick Test Script for BraTS ResUpNet Setup
Run this to verify everything works before full training
"""

import sys
import os

print("="*70)
print("🧪 BraTS ResUpNet - Complete Setup Verification")
print("="*70)

# Step 0: Check Python version
print("\n0️⃣ Checking Python version...")
py_version = sys.version_info
if py_version.major >= 3 and py_version.minor >= 8:
    print(f"   ✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
else:
    print(f"   ❌ Python {py_version.major}.{py_version.minor} (need 3.8+)")
    sys.exit(1)

# Step 1: Check dependencies
print("\n1️⃣ Checking core dependencies...")
try:
    import numpy as np
    import nibabel as nib
    import cv2
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    print("   ✅ All core dependencies installed")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")
    print("   Run: pip install -r requirements_brats.txt")
    sys.exit(1)

# Step 1.5: Check GPU/TensorFlow
print("\n1.5️⃣ Checking GPU & TensorFlow...")
print(f"   TensorFlow version: {tf.__version__}")
print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"   ✅ GPU detected: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"      - {gpu.name}")
    
    # Quick GPU test
    try:
        with tf.device("/GPU:0"):
            a = tf.random.uniform((512, 512))
            b = tf.random.uniform((512, 512))
            c = tf.matmul(a, b)
            _ = c.numpy()
        print("   ✅ GPU computation test passed")
    except Exception as e:
        print(f"   ⚠️ GPU test failed: {e}")
else:
    print("   ⚠️ No GPU detected - will use CPU (slower)")
    print("   For GPU support, install: CUDA 11.8+ and cuDNN 8.6+")

# Step 2: Check if brats_dataloader.py exists
print("\n2️⃣ Checking data loader script...")
if os.path.exists('brats_dataloader.py'):
    print("   ✅ brats_dataloader.py found")
    from brats_dataloader import BraTSDataLoader, save_preprocessed_splits
else:
    print("   ❌ brats_dataloader.py not found in current directory")
    sys.exit(1)

# Step 3: Let user specify BraTS dataset path
print("\n3️⃣ Locating BraTS dataset...")
print("\nPlease specify your BraTS dataset path:")
print("Example: C:/Users/KIIT/Desktop/Datasets/BraTS2021_Training_Data")

default_path = "C:/Users/KIIT/Desktop/Datasets/BraTS2021_Training_Data"
BRATS_ROOT = input(f"Path (press Enter for default: {default_path}): ").strip()

if not BRATS_ROOT:
    BRATS_ROOT = default_path

if not os.path.exists(BRATS_ROOT):
    print(f"   ❌ Directory not found: {BRATS_ROOT}")
    print("\n   Download BraTS dataset first:")
    print("   - Kaggle: kaggle datasets download -d awsaf49/brats2020-training-data")
    print("   - Or see BRATS_QUICKSTART.md for other options")
    sys.exit(1)

print(f"   ✅ Found: {BRATS_ROOT}")

# Step 4: Test data loader with 5 patients
print("\n4️⃣ Testing data loader (5 patients)...")
try:
    loader = BraTSDataLoader(
        dataset_root=BRATS_ROOT,
        modality='flair',
        img_size=(256, 256),
        binary_segmentation=True,
        min_tumor_pixels=50,
        clip_percentile=99.5
    )
    
    print(f"   ✅ Found {len(loader.patient_folders)} patient folders")
    
    # Load 5 patients for quick test
    images, masks, patient_info = loader.load_dataset(max_patients=5, verbose=True)
    
    print(f"\n   📊 Quick Test Results:")
    print(f"      - Total slices: {len(images)}")
    print(f"      - Image shape: {images.shape}")
    print(f"      - Mask shape: {masks.shape}")
    print(f"      - Tumor prevalence: {masks.mean():.4f}")
    print(f"      - Min pixel value: {images.min():.4f}")
    print(f"      - Max pixel value: {images.max():.4f}")
    
    # Check for issues
    if len(images) == 0:
        print("   ❌ No slices extracted! Check dataset structure.")
        sys.exit(1)
    
    if masks.mean() < 0.001:
        print("   ⚠️ Very low tumor prevalence - may indicate loading issue")
    
    print("   ✅ Data loading successful")
    
except Exception as e:
    print(f"   ❌ Error during data loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test train/val/test split
print("\n5️⃣ Testing data split...")
try:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_dataset(
        images, masks, patient_info,
        patient_wise=True,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"   ✅ Data split successful")
    print(f"      Train: {X_train.shape[0]} slices")
    print(f"      Val:   {X_val.shape[0]} slices")
    print(f"      Test:  {X_test.shape[0]} slices")
    
except Exception as e:
    print(f"   ❌ Error during split: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Visualize samples
print("\n6️⃣ Generating visualization...")
try:
    loader.visualize_samples(X_train, y_train, n_samples=4, save_path='test_samples_brats.png')
    print("   ✅ Visualization saved: test_samples_brats.png")
except Exception as e:
    print(f"   ⚠️ Visualization failed (non-critical): {e}")

# Step 7: Test saving
print("\n7️⃣ Testing data save/load...")
try:
    test_dir = 'processed_splits_brats_test'
    save_preprocessed_splits(
        X_train, y_train, X_val, y_val, X_test, y_test,
        output_dir=test_dir
    )
    print(f"   ✅ Data saved to: {test_dir}/")
    
    # Verify files exist
    files = os.listdir(test_dir)
    expected = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
    if all(f in files for f in expected):
        print("   ✅ All split files present")
    else:
        print("   ⚠️ Some split files missing")
    
    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(test_dir, f)) for f in expected)
    print(f"   💾 Total size: {total_size / 1e6:.1f} MB")
    
except Exception as e:
    print(f"   ❌ Error during save: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 8: Check TensorFlow/GPU
print("\n8️⃣ Checking TensorFlow configuration...")
try:
    print(f"   TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"      - {gpu.name}")
    else:
        print("   ℹ️ No GPU detected - will use CPU")
        print("      (Training will be slower but still works)")
except Exception as e:
    print(f"   ⚠️ TensorFlow check failed: {e}")

# Final summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED - System Ready!")
print("="*70)
print("\n📚 Next Steps:")
print("   1. Process full dataset (remove max_patients=5 limit)")
print("   2. Train ResUpNet model (use your existing model code)")
print("   3. Find optimal threshold (run threshold_optimizer.py)")
print("   4. Evaluate final metrics")
print("\n📖 See BRATS_QUICKSTART.md for detailed instructions")
print("="*70)
