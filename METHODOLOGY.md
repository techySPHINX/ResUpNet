# Research Methodology

## 1. Study Overview

### 1.1 Research Objectives

This study investigates the application of ResUpNet architecture for automated brain tumor segmentation using multimodal MRI scans from the BraTS (Brain Tumor Segmentation) dataset. The primary objectives are:

1. **Develop** a robust deep learning model for binary tumor segmentation (tumor vs. background)
2. **Implement** patient-wise data splitting to prevent data leakage and ensure generalizability
3. **Optimize** segmentation threshold to balance clinical precision and recall requirements
4. **Evaluate** model performance using medical imaging metrics (Dice, HD95, ASD)
5. **Validate** reproducibility through fixed random seeds and deterministic operations

### 1.2 Clinical Significance

Automated brain tumor segmentation aims to:

- Reduce manual annotation time for radiologists
- Improve inter-observer consistency in tumor delineation
- Enable quantitative tumor volume tracking for treatment monitoring
- Support surgical planning and radiation therapy targeting

---

## 2. Dataset Description

### 2.1 BraTS Dataset Characteristics

| Attribute       | Description                                   |
| --------------- | --------------------------------------------- |
| **Source**      | Multimodal Brain Tumor Segmentation Challenge |
| **Format**      | NIfTI (.nii.gz)                               |
| **Modalities**  | T1, T1ce (contrast-enhanced), T2, FLAIR       |
| **Resolution**  | 240×240×155 voxels (1mm³ isotropic)           |
| **Tumor Types** | Glioblastoma (HGG), Lower-grade glioma (LGG)  |
| **Annotations** | Expert-verified manual segmentations          |
| **Labels**      | 0=Background, 1=NCR/NET, 2=ED, 4=ET           |

### 2.2 Data Preprocessing Pipeline

#### 2.2.1 Multi-Modal Image Loading

```
For each patient:
  1. Load 4 MRI modalities (T1, T1ce, T2, FLAIR)
  2. Stack modalities → (H, W, D, 4) tensor
  3. Load ground truth segmentation → (H, W, D)
```

#### 2.2.2 Slice Extraction Strategy

- **Extract 2D axial slices** from 3D volumes
- **Filter empty slices**: Remove slices with <1% tumor pixels
- **Rationale**: Reduce class imbalance and training inefficiency

#### 2.2.3 Normalization Strategy

**Patient-wise Z-score normalization** (per modality, per patient):

$$
x_{norm} = \frac{x - \mu_{patient}}{\sigma_{patient} + \epsilon}
$$

Where:

- $\mu_{patient}$ = Mean intensity across patient's entire volume
- $\sigma_{patient}$ = Standard deviation across patient's volume
- $\epsilon = 10^{-8}$ (numerical stability)

**Justification**: Patient-wise normalization prevents information leakage and accounts for inter-patient intensity variations due to scanner/protocol differences.

#### 2.2.4 Binary Label Conversion

Convert multi-class BraTS labels to binary:

```python
binary_mask = (segmentation > 0).astype(np.uint8)
# 0 = Background, 1 = Any tumor region (NCR/NET/ED/ET)
```

---

## 3. Data Splitting Strategy

### 3.1 Patient-Level Stratification

**Critical for Medical AI**: Split by patient ID, NOT by slices

```
1. Group all slices by patient ID
2. Calculate per-patient slice counts
3. Stratified split: 70% Train / 15% Validation / 15% Test
4. Ensure no patient appears in multiple splits
```

**Prevents Data Leakage**: Adjacent slices from the same patient are highly correlated. Random slice-level splitting leads to overoptimistic performance estimates.

### 3.2 Split Statistics Documentation

Document for each split:

- Number of patients
- Number of slices
- Tumor-to-background ratio
- Tumor size distribution (small/medium/large)

---

## 4. Data Augmentation

### 4.1 Training Augmentations

Applied probabilistically during training (p=0.5):

| Augmentation        | Parameters | Rationale                      |
| ------------------- | ---------- | ------------------------------ |
| **Horizontal Flip** | Axis=1     | Anatomical symmetry            |
| **Vertical Flip**   | Axis=0     | Positional invariance          |
| **Rotation**        | ±15°       | Scanner positioning variations |
| **Zoom**            | 0.9-1.1x   | Tumor size variability         |
| **Brightness**      | ±10%       | Scanner intensity variations   |
| **Contrast**        | ±10%       | Protocol-dependent contrast    |

### 4.2 Validation/Test Protocol

- **NO augmentation** during validation/testing
- Ensures unbiased performance evaluation

---

## 5. Model Architecture

### 5.1 ResUpNet Design Philosophy

**Encoder-Decoder with Residual Connections**:

- **Encoder**: 4 residual blocks (16→32→64→128 filters)
- **Bottleneck**: 256 filters
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Single-channel sigmoid activation

### 5.2 Architectural Components

#### Residual Block

```
Input → Conv(3×3) → BatchNorm → ReLU → Conv(3×3) → BatchNorm
  ↓                                                       ↓
  └─────────────────── Identity/Projection ──────────────┘
                              ↓
                           ReLU → Output
```

#### Skip Connections

- **Purpose**: Preserve spatial details from encoder
- **Implementation**: Concatenation (not addition)
- **Medical relevance**: Critical for boundary precision

### 5.3 Regularization Techniques

| Technique               | Configuration          | Purpose              |
| ----------------------- | ---------------------- | -------------------- |
| **Dropout**             | 0.3 (after each block) | Prevent overfitting  |
| **L2 Regularization**   | λ=1×10⁻⁵               | Weight decay         |
| **Batch Normalization** | After each conv layer  | Training stability   |
| **Early Stopping**      | Patience=15 epochs     | Prevent overtraining |

---

## 6. Training Protocol

### 6.1 Loss Function

**Combined Dice + Binary Cross-Entropy Loss**:

$$
\mathcal{L}_{total} = \mathcal{L}_{Dice} + \mathcal{L}_{BCE}
$$

Where:

$$
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}
$$

$$
\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i} [g_i \log(p_i) + (1-g_i) \log(1-p_i)]
$$

- $p_i$ = Predicted probability
- $g_i$ = Ground truth binary label
- $\epsilon = 1$ (smooth Dice coefficient)

### 6.2 Optimization Configuration

| Hyperparameter      | Value                   | Justification              |
| ------------------- | ----------------------- | -------------------------- |
| **Optimizer**       | Adam                    | Adaptive learning rate     |
| **Initial LR**      | 1×10⁻⁴                  | Common for medical imaging |
| **LR Schedule**     | ReduceLROnPlateau       | Adaptive reduction         |
| **LR Reduction**    | Factor=0.5, Patience=10 | Conservative decay         |
| **Batch Size**      | 16                      | GPU memory constraint      |
| **Epochs**          | 50 (max)                | With early stopping        |
| **Mixed Precision** | Enabled (if GPU)        | 2× speed improvement       |

### 6.3 Reproducibility Measures

```python
# Fixed Random Seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Deterministic Operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

**Note**: Deterministic mode may reduce GPU performance by ~5-10% but ensures exact reproducibility.

---

## 7. Threshold Optimization

### 7.1 Motivation

Default threshold (0.5) may not maximize clinical utility. Optimal threshold balances:

- **High Precision**: Minimize false positives (avoid unnecessary interventions)
- **High Recall**: Minimize false negatives (detect all tumor regions)

### 7.2 Optimization Procedure

```
1. Obtain predicted probabilities on validation set
2. Evaluate thresholds ∈ [0.1, 0.2, ..., 0.9]
3. For each threshold:
   - Compute Dice, Precision, Recall, F1
   - Calculate composite score
4. Select threshold maximizing F1 score
```

### 7.3 Threshold Selection Criterion

**Maximize F1 Score** (harmonic mean of Precision and Recall):

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

**Alternative Criteria** (application-dependent):

- Maximize Dice coefficient
- Constrain Precision ≥ 0.85 and maximize Recall
- Constrain Recall ≥ 0.90 and maximize Precision

---

## 8. Evaluation Metrics

### 8.1 Overlap-Based Metrics

#### Dice Similarity Coefficient (DSC)

$$
DSC = \frac{2 |P \cap G|}{|P| + |G|}
$$

- **Range**: [0, 1], higher is better
- **Clinical interpretation**: Overlap quality between prediction (P) and ground truth (G)

#### Intersection over Union (IoU)

$$
IoU = \frac{|P \cap G|}{|P \cup G|}
$$

- **Range**: [0, 1], higher is better
- **Relation to Dice**: $Dice = \frac{2 \times IoU}{1 + IoU}$

### 8.2 Pixel-wise Classification Metrics

#### Precision (Positive Predictive Value)

$$
Precision = \frac{TP}{TP + FP}
$$

- **Clinical meaning**: Proportion of predicted tumor that is actually tumor
- **High precision**: Few false alarms

#### Recall (Sensitivity)

$$
Recall = \frac{TP}{TP + FN}
$$

- **Clinical meaning**: Proportion of actual tumor detected
- **High recall**: Few missed tumors

#### Specificity

$$
Specificity = \frac{TN}{TN + FP}
$$

- **Clinical meaning**: Proportion of healthy tissue correctly identified

### 8.3 Distance-Based Metrics

#### Hausdorff Distance (HD95)

$$
HD_{95}(P, G) = 95^{th} \text{ percentile of } \max\{\max_{p \in P} d(p, G), \max_{g \in G} d(g, P)\}
$$

- **Unit**: Pixels (or mm if calibrated)
- **Clinical interpretation**: Maximum boundary error (outlier-robust)
- **Good performance**: HD95 < 5mm

#### Average Surface Distance (ASD)

$$
ASD = \frac{1}{2} \left[ \frac{1}{|S_P|} \sum_{p \in S_P} d(p, S_G) + \frac{1}{|S_G|} \sum_{g \in S_G} d(g, S_P) \right]
$$

- **Unit**: Pixels (or mm)
- **Clinical interpretation**: Average boundary delineation error

---

## 9. Statistical Analysis

### 9.1 Performance Reporting

For each metric, report:

- **Mean ± Standard Deviation** across test set
- **Median [IQR]** (robust to outliers)
- **Min and Max** (performance range)

### 9.2 Confidence Intervals

Bootstrap 95% confidence intervals (1000 iterations):

```
For b = 1 to 1000:
  1. Resample test set with replacement
  2. Compute metric on resampled set
  3. Store metric value
Compute 2.5th and 97.5th percentiles
```

### 9.3 Subgroup Analysis

Stratify results by:

- **Tumor size**: Small (<10cm²), Medium (10-50cm²), Large (>50cm²)
- **Tumor location**: Frontal, Parietal, Temporal, Occipital
- **Tumor grade**: High-grade glioma (HGG) vs. Low-grade glioma (LGG)

---

## 10. Validation & Testing Protocol

### 10.1 Cross-Validation Strategy

**Hold-out validation** (not k-fold):

- Justification: Large dataset (369 patients) provides sufficient validation samples
- Advantage: Faster training, consistent test set for comparison

### 10.2 Testing Procedure

```
1. Load trained model (best validation checkpoint)
2. Load optimal threshold (from validation)
3. For each test patient:
   a. Normalize using patient-wise statistics
   b. Predict slice-by-slice
   c. Apply optimal threshold
   d. Compute per-slice metrics
4. Aggregate metrics across all test slices
5. Compute patient-level metrics (optional)
```

### 10.3 Qualitative Evaluation

Visualize for representative cases:

- **Best case**: Highest Dice score
- **Median case**: 50th percentile Dice
- **Worst case**: Lowest Dice score
- **Edge cases**: Small tumors, large tumors, irregular shapes

Include in visualization:

- Input MRI (all modalities)
- Ground truth segmentation
- Predicted segmentation
- Error map (FP in red, FN in blue)

---

## 11. Limitations & Assumptions

### 11.1 Dataset Limitations

- **Single dataset**: Results may not generalize to other hospitals/scanners
- **Pre-operative MRI only**: Not applicable to intra-operative guidance
- **Expert annotations**: Inter-rater variability not quantified

### 11.2 Model Limitations

- **2D architecture**: Does not exploit full 3D context
- **Binary segmentation**: Does not distinguish tumor sub-regions
- **Fixed input size**: Requires resizing original resolution

### 11.3 Clinical Translation Barriers

- **Regulatory approval**: Requires prospective clinical validation
- **Integration**: Workflow compatibility with PACS systems
- **Interpretability**: Lack of uncertainty quantification

---

## 12. Ethical Considerations

### 12.1 Data Privacy

- All data de-identified (HIPAA/GDPR compliant)
- Patient consent obtained in original studies
- No re-identification risk in published results

### 12.2 Algorithmic Bias

- Assess performance across demographic subgroups (if available)
- Monitor for disparate performance by tumor grade/location
- Report failure modes transparently

### 12.3 Clinical Use Disclaimer

**This model is for research purposes only and not approved for clinical diagnosis or treatment planning.**

---

## 13. Reproducibility Checklist

- [x] Random seeds fixed (numpy, TensorFlow, Python)
- [x] Software versions documented (requirements.txt)
- [x] Hardware specifications reported
- [x] Data preprocessing steps detailed
- [x] Model architecture diagram provided
- [x] Training hyperparameters listed
- [x] Evaluation metrics defined mathematically
- [x] Code publicly available (GitHub)
- [x] Trained model weights shareable (upon request)

---

## 14. Future Work Directions

### 14.1 Architectural Improvements

- **3D ResUpNet**: Exploit volumetric context
- **Attention mechanisms**: Focus on tumor boundaries
- **Multi-scale fusion**: Process multiple resolutions

### 14.2 Multi-Class Segmentation

- Segment tumor sub-regions (NCR, ED, ET separately)
- Enable quantification of tumor heterogeneity

### 14.3 Uncertainty Quantification

- **Bayesian deep learning**: MC Dropout or Deep Ensembles
- **Predictive interval estimation**: Identify uncertain predictions
- **Clinical utility**: Flag cases requiring expert review

### 14.4 External Validation

- Test on datasets from different institutions
- Evaluate generalization to different MRI protocols
- Prospective validation in clinical workflow

### 14.5 Clinical Decision Support

- Integrate tumor growth tracking over time
- Predict treatment response
- Combine with genomic data for personalized medicine

---

## References

For comprehensive references, see:

- BraTS Challenge: [https://www.med.upenn.edu/cbica/brats/](https://www.med.upenn.edu/cbica/brats/)
- ResNet Architecture: He et al., "Deep Residual Learning for Image Recognition" (2016)
- U-Net Architecture: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Dice Loss: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016)

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Authors**: ResUpNet Research Team  
**Contact**: [GitHub Repository](https://github.com/techySPHINX/ResUpNet)
