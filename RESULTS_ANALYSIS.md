# Results Analysis and Reporting Template

> **Instructions**: This template guides you through comprehensive reporting of your ResUpNet training results. Fill in all sections with your actual experimental data. Replace placeholder values with your measurements and add your generated figures.

---

## Executive Summary

**Training Date**: `[YYYY-MM-DD]`  
**Model Version**: `[e.g., ResUpNet-v1.0]`  
**Dataset**: `[BraTS 2020/2021/2022]`  
**Hardware**: `[GPU model, RAM, CPU]`  
**Training Duration**: `[e.g., 3.5 hours]`

### Key Findings (One-line summary for each)

- **Best Dice Score**: `[Fill in your value]` on test set
- **Optimal Threshold**: `[Fill in your value]` (maximizing F1)
- **Model Convergence**: Epoch `[X]` (early stopping)
- **Clinical Relevance**: `[One sentence on clinical applicability]`

---

## 1. Dataset Statistics

### 1.1 Data Split Summary

| Split          | Patients | Slices | Tumor Slices | Empty Slices | Tumor/Background Ratio |
| -------------- | -------- | ------ | ------------ | ------------ | ---------------------- |
| **Training**   | `[X]`    | `[X]`  | `[X]`        | `[X]`        | `[X.XX]`               |
| **Validation** | `[X]`    | `[X]`  | `[X]`        | `[X]`        | `[X.XX]`               |
| **Test**       | `[X]`    | `[X]`  | `[X]`        | `[X]`        | `[X.XX]`               |

**Patient-level split percentages**: Train `[XX]`% / Val `[XX]`% / Test `[XX]`%

### 1.2 Tumor Size Distribution

| Category   | Definition (cm²) | Train | Val   | Test  |
| ---------- | ---------------- | ----- | ----- | ----- |
| **Small**  | < 10 cm²         | `[X]` | `[X]` | `[X]` |
| **Medium** | 10-50 cm²        | `[X]` | `[X]` | `[X]` |
| **Large**  | > 50 cm²         | `[X]` | `[X]` | `[X]` |

### 1.3 Tumor Grade Distribution (if available)

| Grade   | Description       | Train | Val   | Test  |
| ------- | ----------------- | ----- | ----- | ----- |
| **HGG** | High-Grade Glioma | `[X]` | `[X]` | `[X]` |
| **LGG** | Low-Grade Glioma  | `[X]` | `[X]` | `[X]` |

---

## 2. Training Dynamics

### 2.1 Hyperparameters Used

| Hyperparameter              | Value            |
| --------------------------- | ---------------- |
| **Batch Size**              | `[X]`            |
| **Learning Rate**           | `[X.XXe-X]`      |
| **Optimizer**               | `[Adam/SGD/...]` |
| **Loss Function**           | `[Dice+BCE/...]` |
| **Epochs (Max)**            | `[XX]`           |
| **Early Stopping Patience** | `[XX]`           |
| **LR Reduction Factor**     | `[X.X]`          |
| **LR Reduction Patience**   | `[XX]`           |
| **Dropout Rate**            | `[X.X]`          |
| **L2 Regularization**       | `[X.XXe-X]`      |
| **Random Seed**             | `[XX]`           |
| **Mixed Precision**         | `[Yes/No]`       |

### 2.2 Training Convergence

**Training stopped at epoch**: `[XX]` (early stopping triggered / max epochs reached)  
**Best validation epoch**: `[XX]`  
**Training time**: `[X.X]` hours  
**Time per epoch**: `[XX]` minutes

### 2.3 Learning Curves

> **Insert Figure**: `brats_training_curves.png`  
> Expected plot: Train/Val Loss and Train/Val Dice over epochs

**Observations**:

- Training loss reached minimum at epoch `[XX]`: `[X.XXXX]`
- Validation loss reached minimum at epoch `[XX]`: `[X.XXXX]`
- Generalization gap (Train Dice - Val Dice): `[X.XX]`%
- Evidence of overfitting: `[Yes/No - explain]`

### 2.4 Learning Rate Schedule

> **Insert Figure**: Learning rate vs. epoch (if tracked)

**LR reductions occurred at epochs**: `[e.g., 20, 35, 47]`  
**Final learning rate**: `[X.XXe-X]`

---

## 3. Threshold Optimization Results

### 3.1 Threshold Search Results

> **Insert Figure**: Metrics vs. Threshold plot (0.1 to 0.9)

| Threshold | Dice       | Precision  | Recall     | F1         | IoU        |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 0.1       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.2       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.3       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.4       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| **0.5**   | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.6       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.7       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.8       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |
| 0.9       | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` | `[X.XXXX]` |

### 3.2 Optimal Threshold Selection

**Optimal threshold (maximizing F1)**: `[X.X]`  
**Dice at optimal threshold**: `[X.XXXX]`  
**Precision at optimal threshold**: `[X.XXXX]`  
**Recall at optimal threshold**: `[X.XXXX]`  
**F1 at optimal threshold**: `[X.XXXX]`

**Justification**: `[Explain why this threshold is optimal for your application]`

---

## 4. Test Set Performance

### 4.1 Overall Metrics (Using Optimal Threshold)

| Metric          | Mean ± Std          | Median [IQR]            | Min      | Max      | 95% CI           |
| --------------- | ------------------- | ----------------------- | -------- | -------- | ---------------- |
| **Dice**        | `[X.XXXX ± X.XXXX]` | `[X.XXXX [X.XX, X.XX]]` | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **IoU**         | `[X.XXXX ± X.XXXX]` | `[X.XXXX [X.XX, X.XX]]` | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **Precision**   | `[X.XXXX ± X.XXXX]` | `[X.XXXX [X.XX, X.XX]]` | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **Recall**      | `[X.XXXX ± X.XXXX]` | `[X.XXXX [X.XX, X.XX]]` | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **F1 Score**    | `[X.XXXX ± X.XXXX]` | `[X.XXXX [X.XX, X.XX]]` | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **Specificity** | `[X.XXXX ± X.XXXX]` | `[X.XXXX [X.XX, X.XX]]` | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **HD95 (px)**   | `[X.XX ± X.XX]`     | `[X.XX [X.XX, X.XX]]`   | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |
| **ASD (px)**    | `[X.XX ± X.XX]`     | `[X.XX [X.XX, X.XX]]`   | `[X.XX]` | `[X.XX]` | `[[X.XX, X.XX]]` |

### 4.2 Confusion Matrix Statistics

| Metric              | Value                |
| ------------------- | -------------------- |
| **True Positives**  | `[X,XXX,XXX]` pixels |
| **False Positives** | `[X,XXX,XXX]` pixels |
| **True Negatives**  | `[X,XXX,XXX]` pixels |
| **False Negatives** | `[X,XXX,XXX]` pixels |
| **Total Pixels**    | `[X,XXX,XXX]` pixels |

> **Insert Figure**: `brats_confusion_matrix.png`

---

## 5. Subgroup Analysis

### 5.1 Performance by Tumor Size

| Size Category          | N (slices) | Dice (Mean±Std) | Precision | Recall   | HD95 (px) |
| ---------------------- | ---------- | --------------- | --------- | -------- | --------- |
| **Small (<10 cm²)**    | `[XXX]`    | `[X.XX±X.XX]`   | `[X.XX]`  | `[X.XX]` | `[X.XX]`  |
| **Medium (10-50 cm²)** | `[XXX]`    | `[X.XX±X.XX]`   | `[X.XX]`  | `[X.XX]` | `[X.XX]`  |
| **Large (>50 cm²)**    | `[XXX]`    | `[X.XX±X.XX]`   | `[X.XX]`  | `[X.XX]` | `[X.XX]`  |

**Statistical test**: `[ANOVA/Kruskal-Wallis]`  
**P-value**: `[X.XXX]`  
**Conclusion**: `[Significant/No significant difference between groups]`

> **Insert Figure**: `brats_metrics_distribution.png` (box plots by size)

### 5.2 Performance by Tumor Grade (if available)

| Grade   | N (patients) | Dice (Mean±Std) | Precision | Recall   |
| ------- | ------------ | --------------- | --------- | -------- |
| **HGG** | `[XX]`       | `[X.XX±X.XX]`   | `[X.XX]`  | `[X.XX]` |
| **LGG** | `[XX]`       | `[X.XX±X.XX]`   | `[X.XX]`  | `[X.XX]` |

**Statistical test**: `[t-test/Mann-Whitney U]`  
**P-value**: `[X.XXX]`

### 5.3 Performance by Tumor Location (if annotated)

| Location      | N (slices) | Dice (Mean±Std) |
| ------------- | ---------- | --------------- |
| **Frontal**   | `[XXX]`    | `[X.XX±X.XX]`   |
| **Parietal**  | `[XXX]`    | `[X.XX±X.XX]`   |
| **Temporal**  | `[XXX]`    | `[X.XX±X.XX]`   |
| **Occipital** | `[XXX]`    | `[X.XX±X.XX]`   |

---

## 6. Error Analysis

### 6.1 Failure Mode Analysis

**Best Case** (Highest Dice):

- Dice Score: `[X.XXXX]`
- Patient ID: `[BraTS2021_XXXXX]`
- Tumor characteristics: `[Size, location, grade]`

> **Insert Figure**: Best case visualization

**Median Case**:

- Dice Score: `[X.XXXX]`
- Patient ID: `[BraTS2021_XXXXX]`

> **Insert Figure**: Median case visualization

**Worst Case** (Lowest Dice):

- Dice Score: `[X.XXXX]`
- Patient ID: `[BraTS2021_XXXXX]`
- Tumor characteristics: `[Size, location, grade]`
- Hypothesized reason for failure: `[e.g., very small tumor, boundary ambiguity]`

> **Insert Figure**: Worst case visualization

### 6.2 Systematic Error Patterns

> **Insert Figure**: `brats_error_analysis.png` (FP/FN heatmap)

**False Positive Patterns**:

1. `[e.g., Edge artifacts near skull]`
2. `[e.g., Misclassification of edema]`
3. `[Other observed patterns]`

**False Negative Patterns**:

1. `[e.g., Small isolated tumor regions (<5px)]`
2. `[e.g., Low-contrast tumor boundaries]`
3. `[Other observed patterns]`

---

## 7. Visualization Gallery

### 7.1 Qualitative Segmentation Results

> **Insert Figure**: `brats_qualitative_results.png`  
> Should show: T1, T1ce, T2, FLAIR, Ground Truth, Prediction, Error Map

**Figure caption**: Representative segmentation results showing input modalities, ground truth, model prediction, and error map (red=false positive, blue=false negative).

### 7.2 ROC and Precision-Recall Curves

> **Insert Figure**: `brats_roc_pr_curves.png`

| Metric      | Value      |
| ----------- | ---------- |
| **ROC AUC** | `[X.XXXX]` |
| **PR AUC**  | `[X.XXXX]` |

### 7.3 Metric Correlation Analysis

> **Insert Figure**: `brats_metric_correlation.png` (correlation heatmap)

**Strong correlations observed**:

- Dice vs. IoU: `[r = X.XX]` (expected due to mathematical relationship)
- Precision vs. Recall: `[r = X.XX]`
- Dice vs. HD95: `[r = X.XX]`

### 7.4 Bland-Altman Analysis (Agreement)

> **Insert Figure**: `brats_bland_altman_analysis.png`

**Mean difference (Prediction - Ground Truth)**: `[X.XX]` cm²  
**Limits of agreement**: `[[X.XX, X.XX]]` cm²  
**Interpretation**: `[Most predictions within acceptable range]`

---

## 8. Computational Performance

### 8.1 Training Efficiency

| Metric                   | Value             |
| ------------------------ | ----------------- |
| **Total Training Time**  | `[X.X]` hours     |
| **Time per Epoch**       | `[XX]` minutes    |
| **GPU Utilization**      | `[XX]`% (average) |
| **Peak GPU Memory**      | `[X.X]` GB        |
| **Training Samples/Sec** | `[XXX]`           |

### 8.2 Inference Performance

| Metric                            | Value              |
| --------------------------------- | ------------------ |
| **Inference Time (single image)** | `[XX]` ms          |
| **Throughput (batch=16)**         | `[XXX]` images/sec |
| **GPU Memory (inference)**        | `[XXX]` MB         |
| **Model Size (FP32)**             | `[XX.X]` MB        |
| **Model Size (FP16)**             | `[XX.X]` MB        |

---

## 9. Comparison with Baseline Methods

### 9.1 Literature Comparison

| Method              | Dataset    | Dice     | Precision | Recall   | HD95     | Reference          |
| ------------------- | ---------- | -------- | --------- | -------- | -------- | ------------------ |
| **ResUpNet (Ours)** | BraTS 2021 | `[X.XX]` | `[X.XX]`  | `[X.XX]` | `[X.XX]` | This work          |
| U-Net               | BraTS 2020 | 0.853    | 0.832     | 0.876    | 6.2      | [Ronneberger 2015] |
| Attention U-Net     | BraTS 2020 | 0.871    | 0.854     | 0.889    | 5.8      | [Oktay 2018]       |
| V-Net               | BraTS 2019 | 0.842    | 0.821     | 0.865    | 7.1      | [Milletari 2016]   |

**Statistical significance**: `[Perform paired t-test if using same test set]`

### 9.2 Ablation Study Results (if conducted)

| Model Variant            | Dice     | Precision | Recall   | Notes              |
| ------------------------ | -------- | --------- | -------- | ------------------ |
| **Full ResUpNet**        | `[X.XX]` | `[X.XX]`  | `[X.XX]` | All components     |
| w/o Residual Connections | `[X.XX]` | `[X.XX]`  | `[X.XX]` | -X.XX Dice         |
| w/o Skip Connections     | `[X.XX]` | `[X.XX]`  | `[X.XX]` | -X.XX Dice         |
| w/o Dropout              | `[X.XX]` | `[X.XX]`  | `[X.XX]` | Overfitting        |
| w/o Batch Norm           | `[X.XX]` | `[X.XX]`  | `[X.XX]` | Slower convergence |

---

## 10. Clinical Relevance Assessment

### 10.1 Clinical Metrics Summary

| Clinical Criterion           | Target       | Achieved  | Status |
| ---------------------------- | ------------ | --------- | ------ |
| **Tumor Detection Rate**     | >95%         | `[XX]`%   | ✅/❌  |
| **False Alarm Rate**         | <5%          | `[XX]`%   | ✅/❌  |
| **Boundary Accuracy (HD95)** | <5mm         | `[X.X]`mm | ✅/❌  |
| **Processing Time**          | <60s/patient | `[XX]`s   | ✅/❌  |

### 10.2 Clinical Use Case Readiness

**Screening**: `[Ready/Needs improvement - explain]`  
**Treatment Planning**: `[Ready/Needs improvement - explain]`  
**Follow-up Monitoring**: `[Ready/Needs improvement - explain]`

### 10.3 Expert Radiologist Comparison (if available)

| Evaluator                 | Dice     | Inter-rater Agreement |
| ------------------------- | -------- | --------------------- |
| **ResUpNet**              | `[X.XX]` | -                     |
| **Radiologist 1**         | `[X.XX]` | -                     |
| **Radiologist 2**         | `[X.XX]` | -                     |
| **Radiologist Consensus** | `[X.XX]` | `[X.XX (kappa)]`      |

---

## 11. Limitations and Future Work

### 11.1 Current Limitations

1. **Dataset Limitations**:
   - `[e.g., Single institution, limited diversity]`
2. **Model Limitations**:
   - `[e.g., 2D architecture, no uncertainty quantification]`
3. **Performance Gaps**:
   - `[e.g., Struggles with very small tumors (<5mm)]`

### 11.2 Recommended Improvements

**Short-term** (next 3 months):

- `[e.g., Implement 3D architecture]`
- `[e.g., Add multi-scale fusion]`

**Long-term** (next year):

- `[e.g., Multi-center validation study]`
- `[e.g., Prospective clinical trial]`

---

## 12. Reproducibility Information

### 12.1 Software Environment

```yaml
Python: [X.X.X]
TensorFlow: [X.XX.X]
CUDA: [XX.X]
cuDNN: [X.X]
NumPy: [X.XX.X]
Matplotlib: [X.X.X]
scikit-learn: [X.XX.X]
nibabel: [X.X.X]
```

### 12.2 Hardware Specifications

```
GPU: [e.g., NVIDIA RTX 3080, 10GB VRAM]
CPU: [e.g., Intel i7-10700K, 8 cores]
RAM: [e.g., 32GB DDR4]
OS: [e.g., Windows 11 / Ubuntu 20.04]
```

### 12.3 Random Seeds

```python
RANDOM_SEED = [XX]
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
```

### 12.4 Model Checkpoint

**Saved model path**: `[path/to/best_model.h5]`  
**Optimal threshold**: `[X.X]`  
**Epoch**: `[XX]`  
**Validation Dice**: `[X.XXXX]`

---

## 13. Conclusions

### 13.1 Summary of Findings

`[Write 3-5 bullet points summarizing key results]`

- Achieved Dice coefficient of `[X.XX]`, `[exceeding/comparable to]` state-of-the-art
- Optimal threshold of `[X.X]` balances precision (`[X.XX]`) and recall (`[X.XX]`)
- Model demonstrates `[strong/moderate/weak]` generalization (train-test gap: `[X.XX]`)
- `[Other key findings]`

### 13.2 Research Contributions

1. `[e.g., Demonstrated ResUpNet efficacy on BraTS dataset]`
2. `[e.g., Novel threshold optimization approach]`
3. `[e.g., Comprehensive error analysis revealing...]`

### 13.3 Clinical Impact Statement

`[One paragraph on potential clinical utility, limitations for deployment, and next steps for translation]`

---

## 14. Appendices

### Appendix A: Full Hyperparameter Grid Search (if conducted)

`[Table of all tested configurations and results]`

### Appendix B: Per-Patient Results

`[Table with Dice, Precision, Recall for each test patient]`

### Appendix C: Additional Visualizations

`[Any supplementary figures not included in main text]`

---

## References

1. Menze, B. H., et al. (2015). "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE Transactions on Medical Imaging.
2. Bakas, S., et al. (2017). "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features." Scientific Data.
3. `[Add your relevant references]`

---

**Report Author**: `[Your Name]`  
**Date**: `[YYYY-MM-DD]`  
**Contact**: `[email@example.com]`  
**GitHub**: `[link to repository]`  
**DOI** (if published): `[10.XXXX/...]`

---

**Last Updated**: February 2026  
**Template Version**: 1.0  
**Repository**: [techySPHINX/ResUpNet](https://github.com/techySPHINX/ResUpNet)
