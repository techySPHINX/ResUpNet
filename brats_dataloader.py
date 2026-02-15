"""
BraTS Dataset Loader for Medical Research
Supports BraTS 2020/2021/2023 datasets

Features:
- NIfTI (.nii.gz) file loading
- 2D slice extraction from 3D volumes
- Z-score normalization per patient
- Tumor presence filtering
- Multi-modal support (T1, T1ce, T2, FLAIR)
"""

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class BraTSDataLoader:
    """
    BraTS Dataset Loader for 2D slice-based segmentation
    
    Args:
        dataset_root: Path to BraTS dataset (e.g., 'BraTS2021_Training_Data')
        modality: MRI sequence to use ('t1ce', 'flair', 't2', 't1')
        img_size: Target image size (H, W)
        binary_segmentation: If True, combine all tumor labels into binary mask
        min_tumor_pixels: Minimum tumor pixels to include slice (reduces empty slices)
        clip_percentile: Intensity clipping percentile (reduces outliers)
    """
    
    def __init__(
        self,
        dataset_root,
        modality='flair',  # FLAIR often gives best tumor contrast
        img_size=(256, 256),
        binary_segmentation=True,
        min_tumor_pixels=50,
        clip_percentile=99.5
    ):
        self.dataset_root = dataset_root
        self.modality = modality
        self.img_size = img_size
        self.binary_segmentation = binary_segmentation
        self.min_tumor_pixels = min_tumor_pixels
        self.clip_percentile = clip_percentile
        
        # Find all patient folders
        self.patient_folders = self._find_patient_folders()
        print(f"Found {len(self.patient_folders)} patient scans")
    
    def _find_patient_folders(self):
        """Find all patient folders in dataset"""
        folders = []
        for entry in sorted(os.listdir(self.dataset_root)):
            path = os.path.join(self.dataset_root, entry)
            if os.path.isdir(path):
                # Check if it has required files
                files = os.listdir(path)
                has_modality = any(self.modality in f for f in files)
                has_seg = any('seg' in f for f in files)
                if has_modality and has_seg:
                    folders.append(path)
        return folders
    
    def _load_nifti(self, filepath):
        """Load NIfTI file and return numpy array"""
        nii = nib.load(filepath)
        data = nii.get_fdata()
        return data
    
    def _normalize_volume(self, volume):
        """
        Z-score normalization per volume (patient-wise)
        Clip outliers before normalization for stability
        """
        # Clip extreme values
        if self.clip_percentile < 100:
            upper = np.percentile(volume, self.clip_percentile)
            volume = np.clip(volume, 0, upper)
        
        # Brain mask (non-zero region)
        brain_mask = volume > 0
        
        if brain_mask.sum() == 0:
            return volume
        
        # Z-score normalization on brain region only
        mean = volume[brain_mask].mean()
        std = volume[brain_mask].std()
        
        if std > 0:
            volume = (volume - mean) / std
            # Clip normalized values to reasonable range
            volume = np.clip(volume, -5, 5)
        
        return volume
    
    def _resize_slice(self, slice_2d, is_mask=False):
        """Resize 2D slice to target size"""
        import cv2
        
        if is_mask:
            # Use nearest neighbor for masks to preserve labels
            interpolation = cv2.INTER_NEAREST
        else:
            # Use bilinear for images
            interpolation = cv2.INTER_LINEAR
        
        resized = cv2.resize(slice_2d, self.img_size, interpolation=interpolation)
        return resized
    
    def _process_patient(self, patient_folder):
        """
        Process single patient: load volume and extract 2D slices
        Returns: (image_slices, mask_slices, slice_indices)
        """
        # Find files
        files = os.listdir(patient_folder)
        
        # Find modality file
        modality_file = None
        for f in files:
            if self.modality in f and not 'seg' in f:
                modality_file = os.path.join(patient_folder, f)
                break
        
        # Find segmentation file
        seg_file = None
        for f in files:
            if 'seg' in f:
                seg_file = os.path.join(patient_folder, f)
                break
        
        if not modality_file or not seg_file:
            return [], [], []
        
        # Load volumes
        img_volume = self._load_nifti(modality_file)  # (H, W, D)
        seg_volume = self._load_nifti(seg_file)
        
        # Normalize image volume (patient-wise)
        img_volume = self._normalize_volume(img_volume)
        
        # Process segmentation
        if self.binary_segmentation:
            # Binary: 0 = background, 1 = any tumor
            seg_volume = (seg_volume > 0).astype(np.float32)
        
        # Extract 2D slices (axial view: along depth dimension)
        image_slices = []
        mask_slices = []
        slice_indices = []
        
        depth = img_volume.shape[2]
        
        for z in range(depth):
            img_slice = img_volume[:, :, z]
            mask_slice = seg_volume[:, :, z]
            
            # Filter: only keep slices with sufficient tumor pixels
            tumor_pixels = mask_slice.sum()
            
            if tumor_pixels >= self.min_tumor_pixels:
                # Resize
                img_resized = self._resize_slice(img_slice, is_mask=False)
                mask_resized = self._resize_slice(mask_slice, is_mask=True)
                
                # Ensure mask is binary after resize
                if self.binary_segmentation:
                    mask_resized = (mask_resized > 0.5).astype(np.float32)
                
                # Add channel dimension
                img_resized = np.expand_dims(img_resized, axis=-1)  # (H, W, 1)
                mask_resized = np.expand_dims(mask_resized, axis=-1)
                
                image_slices.append(img_resized)
                mask_slices.append(mask_resized)
                slice_indices.append(z)
        
        return image_slices, mask_slices, slice_indices
    
    def load_dataset(self, max_patients=None, verbose=True):
        """
        Load entire dataset
        
        Args:
            max_patients: Limit number of patients (for testing/debugging)
            verbose: Show progress bar
        
        Returns:
            images: numpy array (N, H, W, 1)
            masks: numpy array (N, H, W, 1)
            patient_info: list of dicts with patient metadata
        """
        all_images = []
        all_masks = []
        patient_info = []
        
        patient_folders = self.patient_folders
        if max_patients:
            patient_folders = patient_folders[:max_patients]
        
        iterator = tqdm(patient_folders) if verbose else patient_folders
        
        for patient_folder in iterator:
            patient_id = os.path.basename(patient_folder)
            
            imgs, msks, slice_ids = self._process_patient(patient_folder)
            
            if len(imgs) > 0:
                all_images.extend(imgs)
                all_masks.extend(msks)
                
                # Store metadata
                for slice_id in slice_ids:
                    patient_info.append({
                        'patient_id': patient_id,
                        'slice_index': slice_id
                    })
        
        images = np.array(all_images, dtype=np.float32)
        masks = np.array(all_masks, dtype=np.float32)
        
        if verbose:
            print(f"\n✅ Dataset loaded successfully!")
            print(f"   Total slices: {len(images)}")
            print(f"   Image shape: {images.shape}")
            print(f"   Mask shape: {masks.shape}")
            print(f"   Tumor prevalence: {masks.mean():.4f}")
        
        return images, masks, patient_info
    
    def split_dataset(
        self,
        images,
        masks,
        patient_info,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        patient_wise=True
    ):
        """
        Split dataset into train/val/test
        
        Args:
            patient_wise: If True, ensure same patient's slices stay together
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        if patient_wise:
            # Group by patient
            patient_ids = [info['patient_id'] for info in patient_info]
            unique_patients = sorted(list(set(patient_ids)))
            
            # Split patients
            train_patients, temp_patients = train_test_split(
                unique_patients,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )
            
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state
            )
            
            # Get indices for each split
            train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
            val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
            test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
            
            X_train, y_train = images[train_indices], masks[train_indices]
            X_val, y_val = images[val_indices], masks[val_indices]
            X_test, y_test = images[test_indices], masks[test_indices]
        else:
            # Random split (simpler but not ideal for medical data)
            X_train, X_temp, y_train, y_temp = train_test_split(
                images, masks,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                shuffle=True
            )
            
            val_size = val_ratio / (val_ratio + test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_size),
                random_state=random_state,
                shuffle=True
            )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Train: {X_train.shape[0]} slices ({len(set([info['patient_id'] for i, info in enumerate(patient_info) if i in train_indices]))} patients)")
        print(f"   Val:   {X_val.shape[0]} slices ({len(set([info['patient_id'] for i, info in enumerate(patient_info) if i in val_indices]))} patients)")
        print(f"   Test:  {X_test.shape[0]} slices ({len(set([info['patient_id'] for i, info in enumerate(patient_info) if i in test_indices]))} patients)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def visualize_samples(self, images, masks, n_samples=4, save_path=None):
        """Visualize random samples from dataset"""
        indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples))
        
        for i, idx in enumerate(indices):
            # Image
            axes[i, 0].imshow(images[idx].squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Sample {idx} - Image')
            axes[i, 0].axis('off')
            
            # Mask
            axes[i, 1].imshow(masks[idx].squeeze(), cmap='gray')
            axes[i, 1].set_title(f'Sample {idx} - Mask')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()


def save_preprocessed_splits(X_train, y_train, X_val, y_val, X_test, y_test, output_dir='processed_splits_brats'):
    """Save preprocessed data splits to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f"{output_dir}/X_train.npy", X_train)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/X_val.npy", X_val)
    np.save(f"{output_dir}/y_val.npy", y_val)
    np.save(f"{output_dir}/X_test.npy", X_test)
    np.save(f"{output_dir}/y_test.npy", y_test)
    
    print(f"\n✅ Data saved to {output_dir}/")


def load_preprocessed_splits(input_dir='processed_splits_brats'):
    """Load preprocessed data splits from disk"""
    X_train = np.load(f"{input_dir}/X_train.npy")
    y_train = np.load(f"{input_dir}/y_train.npy")
    X_val = np.load(f"{input_dir}/X_val.npy")
    y_val = np.load(f"{input_dir}/y_val.npy")
    X_test = np.load(f"{input_dir}/X_test.npy")
    y_test = np.load(f"{input_dir}/y_test.npy")
    
    print(f"\n✅ Data loaded from {input_dir}/")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Example usage
if __name__ == "__main__":
    # Example: Load BraTS2021 dataset
    BRATS_ROOT = "path/to/BraTS2021_Training_Data"
    
    loader = BraTSDataLoader(
        dataset_root=BRATS_ROOT,
        modality='flair',  # Best for tumor visibility
        img_size=(256, 256),
        binary_segmentation=True,
        min_tumor_pixels=50
    )
    
    # Load dataset (use max_patients for quick testing)
    images, masks, patient_info = loader.load_dataset(max_patients=10)
    
    # Split dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_dataset(
        images, masks, patient_info,
        patient_wise=True
    )
    
    # Visualize
    loader.visualize_samples(X_train, y_train, n_samples=4)
    
    # Save for later use
    save_preprocessed_splits(X_train, y_train, X_val, y_val, X_test, y_test)
