# Contributing to ResUpNet

First off, thank you for considering contributing to ResUpNet! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Submission Guidelines](#submission-guidelines)
- [Coding Standards](#coding-standards)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, TensorFlow version, GPU info)
- **Code snippets** or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Rationale** for the enhancement
- **Potential implementation** approach
- **Examples** of how it would work

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** following our coding standards
4. **Test thoroughly**
5. **Commit** with clear messages:
   ```bash
   git commit -m "Add amazing feature: description"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Before/after comparisons (if applicable)

## Development Setup

### Prerequisites

```bash
Python 3.8+
CUDA 11.x (for GPU support)
Git
```

### Installation

1. **Clone your fork**:

   ```bash
   git clone https://github.com/your-username/ResUpNet-feat-brats.git
   cd ResUpNet-feat-brats
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements_brats.txt
   ```

4. **Verify setup**:

   ```bash
   python test_brats_setup.py
   ```

5. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Submission Guidelines

### Commit Messages

Follow conventional commits:

```
feat: Add new attention mechanism
fix: Correct Dice loss calculation
docs: Update README with new examples
style: Format code with black
refactor: Simplify data loading pipeline
test: Add unit tests for metrics
perf: Optimize threshold optimization
```

### Code Review Process

1. Maintainers will review your PR
2. Address feedback and push updates
3. Once approved, your PR will be merged
4. Your contribution will be credited

## Coding Standards

### Python Style

- Follow **PEP 8**
- Use **type hints**
- Write **docstrings** (Google style)
- Format with **black** (line length 100)

### Example

```python
from typing import Tuple
import numpy as np

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                    smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient between ground truth and prediction.

    Args:
        y_true: Ground truth binary mask (H, W)
        y_pred: Predicted binary mask (H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient score (0-1, higher is better)

    Example:
        >>> y_true = np.array([[0, 1], [1, 1]])
        >>> y_pred = np.array([[0, 1], [1, 0]])
        >>> dice = dice_coefficient(y_true, y_pred)
        >>> print(f"Dice: {dice:.4f}")
        Dice: 0.6667
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)

    return (2. * intersection + smooth) / (union + smooth)
```

### Testing

- Add tests for new features
- Ensure existing tests pass
- Use `pytest` for testing

```bash
pytest tests/
```

### Documentation

- Update README.md if adding features
- Update DOCUMENTATION.md for technical changes
- Add docstrings to all functions
- Include usage examples

## Types of Contributions

### 🐛 Bug Fixes

- Fix reported issues
- Add test cases
- Update documentation

### ✨ New Features

- New architectures or components
- Additional metrics
- Enhanced visualizations
- Performance optimizations

### 📚 Documentation

- Improve README
- Add tutorials
- Fix typos
- Add examples

### 🧪 Testing

- Add unit tests
- Integration tests
- Performance benchmarks

### 🎨 Improvements

- Code refactoring
- UI/UX enhancements
- Better error messages

## Community

### Communication Channels

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions

### Recognition

Contributors are recognized in:

- Contributors section of README
- Release notes
- Acknowledgments in publications (for significant contributions)

## Questions?

Feel free to:

- Open an issue with the "question" label
- Start a discussion
- Contact maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to ResUpNet!** 🚀

Your efforts help advance medical AI research and improve brain tumor segmentation for the community.
