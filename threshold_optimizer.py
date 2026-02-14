"""
Threshold Optimization for Medical Image Segmentation

Finds optimal threshold to maximize Dice, F1, or balance Precision/Recall
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from tqdm import tqdm


def dice_score(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def compute_metrics_at_threshold(y_true_all, y_pred_prob_all, threshold):
    """Compute all metrics at a specific threshold"""
    y_pred = (y_pred_prob_all > threshold).astype(np.float32)
    
    # Flatten for pixel-wise metrics
    y_true_flat = y_true_all.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    tp = np.sum(y_true_flat * y_pred_flat)
    fp = np.sum((1 - y_true_flat) * y_pred_flat)
    fn = np.sum(y_true_flat * (1 - y_pred_flat))
    tn = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    dice = dice_score(y_true_flat, y_pred_flat)
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def find_optimal_threshold(
    model,
    X_val,
    y_val,
    thresholds=np.linspace(0.1, 0.9, 81),
    optimize_for='f1',
    verbose=True
):
    """
    Find optimal threshold by grid search on validation set
    
    Args:
        model: Trained model
        X_val: Validation images
        y_val: Validation masks
        thresholds: Array of thresholds to test
        optimize_for: 'f1', 'dice', 'balanced' (precision=recall), or 'youden' (sens+spec-1)
        verbose: Print progress
    
    Returns:
        optimal_threshold: Best threshold value
        results: Dict with metrics at all thresholds
    """
    
    if verbose:
        print(f"🔍 Searching optimal threshold (optimizing for: {optimize_for})")
        print(f"   Testing {len(thresholds)} thresholds...")
    
    # Get predictions (probabilities)
    if verbose:
        print("   Generating predictions...")
    
    y_pred_prob = model.predict(X_val, verbose=0)
    
    # Evaluate each threshold
    results = {
        'thresholds': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': [],
        'youden': []  # Youden's J statistic = sensitivity + specificity - 1
    }
    
    iterator = tqdm(thresholds) if verbose else thresholds
    
    for thresh in iterator:
        metrics = compute_metrics_at_threshold(y_val, y_pred_prob, thresh)
        
        results['thresholds'].append(thresh)
        results['dice'].append(metrics['dice'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
        results['f1'].append(metrics['f1'])
        results['specificity'].append(metrics['specificity'])
        results['youden'].append(metrics['recall'] + metrics['specificity'] - 1)
    
    # Find optimal threshold
    if optimize_for == 'f1':
        optimal_idx = np.argmax(results['f1'])
    elif optimize_for == 'dice':
        optimal_idx = np.argmax(results['dice'])
    elif optimize_for == 'balanced':
        # Minimize difference between precision and recall
        diff = np.abs(np.array(results['precision']) - np.array(results['recall']))
        optimal_idx = np.argmin(diff)
    elif optimize_for == 'youden':
        optimal_idx = np.argmax(results['youden'])
    else:
        raise ValueError(f"Unknown optimization criterion: {optimize_for}")
    
    optimal_threshold = results['thresholds'][optimal_idx]
    
    if verbose:
        print(f"\n✅ Optimal threshold found: {optimal_threshold:.3f}")
        print(f"   Dice: {results['dice'][optimal_idx]:.4f}")
        print(f"   F1: {results['f1'][optimal_idx]:.4f}")
        print(f"   Precision: {results['precision'][optimal_idx]:.4f}")
        print(f"   Recall: {results['recall'][optimal_idx]:.4f}")
        print(f"   Specificity: {results['specificity'][optimal_idx]:.4f}")
    
    return optimal_threshold, results


def plot_threshold_analysis(results, optimal_threshold, save_path='threshold_analysis.png'):
    """
    Plot comprehensive threshold analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    thresholds = results['thresholds']
    
    # Plot 1: All metrics vs threshold
    ax1 = axes[0, 0]
    ax1.plot(thresholds, results['dice'], 'b-', linewidth=2, label='Dice')
    ax1.plot(thresholds, results['f1'], 'g-', linewidth=2, label='F1')
    ax1.plot(thresholds, results['precision'], 'r--', linewidth=1.5, label='Precision')
    ax1.plot(thresholds, results['recall'], 'orange', linestyle='--', linewidth=1.5, label='Recall')
    ax1.axvline(optimal_threshold, color='black', linestyle=':', linewidth=2, label=f'Optimal ({optimal_threshold:.3f})')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Threshold', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Precision-Recall tradeoff
    ax2 = axes[0, 1]
    ax2.plot(results['recall'], results['precision'], 'b-', linewidth=2)
    
    # Mark optimal point
    opt_idx = results['thresholds'].index(optimal_threshold)
    ax2.plot(results['recall'][opt_idx], results['precision'][opt_idx], 
             'r*', markersize=20, label=f'Optimal (T={optimal_threshold:.3f})')
    
    # Add threshold annotations at key points
    for i in range(0, len(thresholds), 15):
        ax2.annotate(f'{thresholds[i]:.2f}', 
                    (results['recall'][i], results['precision'][i]),
                    fontsize=8, alpha=0.6)
    
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Tradeoff', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Dice and F1 comparison
    ax3 = axes[1, 0]
    ax3.plot(thresholds, results['dice'], 'b-', linewidth=2, label='Dice')
    ax3.plot(thresholds, results['f1'], 'g-', linewidth=2, label='F1')
    ax3.axvline(optimal_threshold, color='black', linestyle=':', linewidth=2, label=f'Optimal')
    ax3.fill_between(thresholds, results['dice'], results['f1'], alpha=0.2)
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Dice vs F1 Score', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Youden's Index (Sensitivity + Specificity - 1)
    ax4 = axes[1, 1]
    ax4.plot(thresholds, results['youden'], 'purple', linewidth=2, label="Youden's J")
    ax4.plot(thresholds, results['recall'], 'orange', linestyle='--', alpha=0.7, label='Sensitivity')
    ax4.plot(thresholds, results['specificity'], 'cyan', linestyle='--', alpha=0.7, label='Specificity')
    ax4.axvline(optimal_threshold, color='black', linestyle=':', linewidth=2, label=f'Optimal')
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title("Youden's Index (Sensitivity + Specificity - 1)", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Threshold analysis plot saved: {save_path}")
    plt.show()


def compare_thresholds(model, X_test, y_test, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Compare performance at multiple thresholds
    """
    print(f"\n📊 Comparing {len(thresholds)} thresholds on test set:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Dice':<8} {'F1':<8} {'Precision':<12} {'Recall':<12} {'Specificity':<12}")
    print("-" * 80)
    
    y_pred_prob = model.predict(X_test, verbose=0)
    
    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(y_test, y_pred_prob, thresh)
        
        print(f"{thresh:<12.2f} {metrics['dice']:<8.4f} {metrics['f1']:<8.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['specificity']:<12.4f}")
    
    print("-" * 80)


def find_per_sample_optimal_threshold(y_true, y_pred_prob, metric='f1'):
    """
    Find optimal threshold for a single sample
    
    Useful for adaptive thresholding strategies
    """
    best_score = 0
    best_thresh = 0.5
    
    for thresh in np.linspace(0.1, 0.9, 41):
        y_pred = (y_pred_prob > thresh).astype(np.float32)
        
        if metric == 'f1':
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            tp = np.sum(y_true_flat * y_pred_flat)
            fp = np.sum((1 - y_true_flat) * y_pred_flat)
            fn = np.sum(y_true_flat * (1 - y_pred_flat))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            score = 2 * precision * recall / (precision + recall + 1e-8)
        elif metric == 'dice':
            score = dice_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score


# Example usage
if __name__ == "__main__":
    import tensorflow as tf
    
    # Load your trained model
    model = tf.keras.models.load_model('best_resupnet.keras')
    
    # Load validation data
    X_val = np.load('processed_splits/X_val.npy')
    y_val = np.load('processed_splits/y_val.npy')
    
    # Find optimal threshold
    optimal_threshold, results = find_optimal_threshold(
        model, X_val, y_val,
        optimize_for='f1',  # or 'dice', 'balanced', 'youden'
        verbose=True
    )
    
    # Plot analysis
    plot_threshold_analysis(results, optimal_threshold)
    
    # Compare multiple thresholds
    X_test = np.load('processed_splits/X_test.npy')
    y_test = np.load('processed_splits/y_test.npy')
    
    compare_thresholds(model, X_test, y_test)
