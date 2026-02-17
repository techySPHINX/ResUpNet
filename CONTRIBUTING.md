# Contributing to ResUpNet Research Project

Thank you for your interest in contributing to the ResUpNet brain tumor segmentation project! This document provides guidelines for research collaborators, developers, and the broader medical AI community.

---

## 🎯 Types of Contributions

We welcome several types of contributions:

### 🔬 Research Contributions

- **Experimental results**: Share your training results using [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)
- **Ablation studies**: Test different architectural variants
- **Dataset extensions**: Validate on other brain tumor datasets (e.g., BraTS 2022, 2023)
- **Clinical validation**: Collaborate with radiologists for expert evaluation

### 💻 Code Contributions

- **Bug fixes**: Improve code reliability
- **Feature additions**: New metrics, visualizations, or model components
- **Performance optimizations**: Speed or memory improvements
- **Documentation**: Clarify existing docs or add tutorials

### 📊 Benchmark Contributions

- **Baseline comparisons**: Implement and compare with other architectures
- **Hyperparameter tuning**: Share optimal configurations
- **Hardware benchmarks**: Report performance on different GPUs/TPUs

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR-USERNAME/ResUpNet.git
cd ResUpNet
git remote add upstream https://github.com/techySPHINX/ResUpNet.git
```

### 2. Create a Branch

```bash
# For features
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/bug-description

# For documentation
git checkout -b docs/improvement-description
```

### 3. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements_brats.txt
pip install -r requirements_dev.txt  # If available
```

---

## 📝 Contribution Guidelines

### Code Quality

#### Python Code Style

- Follow **PEP 8** style guide
- Use meaningful variable names (e.g., `dice_coefficient` not `dc`)
- Add docstrings to all functions:
  ```python
  def calculate_dice(y_true, y_pred, smooth=1.0):
      """
      Calculate Dice similarity coefficient.

      Args:
          y_true (np.ndarray): Ground truth binary mask
          y_pred (np.ndarray): Predicted binary mask
          smooth (float): Smoothing factor (default: 1.0)

      Returns:
          float: Dice coefficient [0, 1]
      """
      intersection = np.sum(y_true * y_pred)
      return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
  ```

#### Notebook Style

- **Clear cell organization**: One concept per cell
- **Markdown headers**: Use `#`, `##`, `###` for structure
- **Comments**: Explain "why", not just "what"
- **Output**: Clear the output before committing (to reduce file size)
- **Reproducibility**: Set random seeds at the beginning

### Testing

Before submitting:

```bash
# Test environment setup
python test_brats_setup.py

# Run notebook (if code changes affect it)
jupyter nbconvert --to notebook --execute resunet_brats_medical.ipynb --output test_output.ipynb

# Clean up
rm test_output.ipynb
```

### Documentation

- **Update README.md** if adding major features
- **Update METHODOLOGY.md** if changing training protocol
- **Update ARCHITECTURE.md** if modifying model structure
- **Add inline comments** for complex algorithms
- **Include references** for scientific methods

---

## 🔬 Research Contribution Workflow

### Sharing Experimental Results

1. **Run your experiments** using the notebook
2. **Fill out [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)** template
3. **Create a new branch**: `results/your-experiment-name`
4. **Rename your results file**: `RESULTS_ANALYSIS_your_name_date.md`
5. **Include all generated figures** in a folder: `results/your_name_date/`
6. **Commit and push**:
   ```bash
   git add RESULTS_ANALYSIS_your_name_date.md
   git add results/your_name_date/
   git commit -m "Add experimental results: [brief description]"
   git push origin results/your-experiment-name
   ```
7. **Open a Pull Request** with detailed description

### Information to Include

When sharing research results:

- [ ] Hardware used (GPU model, RAM)
- [ ] Software versions (Python, TensorFlow, CUDA)
- [ ] Dataset version (BraTS 2020/2021/2022)
- [ ] Data split (number of patients in train/val/test)
- [ ] Hyperparameters (batch size, epochs, learning rate)
- [ ] Training time
- [ ] Performance metrics (Dice, Precision, Recall, etc.)
- [ ] Visualizations (training curves, segmentation examples)
- [ ] Statistical analysis (CI, subgroup results)

---

## 🐛 Bug Reports

### Before Submitting

1. **Check existing issues**: Search GitHub Issues to avoid duplicates
2. **Verify it's a bug**: Test on a clean environment
3. **Minimal reproducible example**: Isolate the problem

### Bug Report Template

```markdown
**Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:

1. Run cell X in notebook
2. Execute command Y
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Error Message**
```

[Paste full error traceback]

```

**Environment**
- OS: [e.g., Windows 11, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- TensorFlow: [e.g., 2.13.0]
- GPU: [e.g., RTX 3080, or CPU only]
- CUDA: [e.g., 11.8]

**Additional Context**
Any other relevant information.
```

---

## ✨ Feature Requests

### Before Submitting

1. **Check documentation**: Feature may already exist
2. **Review issues**: Someone may have suggested it
3. **Consider scope**: Should align with project goals (medical AI research)

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why is this feature valuable? What problem does it solve?

**Proposed Implementation**
How should this be implemented (optional)?

**Alternatives Considered**
Other approaches you've thought about (optional).

**Research Justification**
Link to papers or clinical evidence supporting this feature (if applicable).
```

---

## 🔄 Pull Request Process

### Before Submitting PR

- [ ] Code follows project style guidelines
- [ ] All tests pass (`python test_brats_setup.py`)
- [ ] Documentation updated (README, METHODOLOGY, etc.)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up-to-date with `main`:
  ```bash
  git fetch upstream
  git rebase upstream/main
  ```

### PR Title Format

- **Feature**: `feat: Add 3D convolution support`
- **Bug fix**: `fix: Correct Dice loss calculation`
- **Documentation**: `docs: Update METHODOLOGY with new metrics`
- **Performance**: `perf: Optimize data loading pipeline`
- **Research**: `research: Add results for BraTS 2022`

### PR Description Template

```markdown
## Description

Brief summary of changes.

## Motivation

Why is this change needed?

## Changes Made

- Bullet point 1
- Bullet point 2

## Testing

How was this tested?

- [ ] Ran notebook end-to-end
- [ ] Verified reproducibility
- [ ] Checked performance metrics

## Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)

[Add training curves, segmentation results, etc.]
```

---

## 📊 Dataset Contributions

### Extending to Other Datasets

If you've adapted ResUpNet for other brain tumor datasets:

1. **Create a new branch**: `dataset/dataset-name`
2. **Add a new notebook**: `resunet_[dataset]_medical.ipynb`
3. **Document differences** in preprocessing, labels, metrics
4. **Update README** with dataset comparison table
5. **Include benchmark results**

### Dataset Comparison Table (example)

| Dataset       | Modalities       | Patients | Resolution  | Labels    | Dice (Ours) |
| ------------- | ---------------- | -------- | ----------- | --------- | ----------- |
| BraTS 2021    | T1/T1ce/T2/FLAIR | 369      | 240×240×155 | 4 classes | 0.XX        |
| [New Dataset] | [...]            | [...]    | [...]       | [...]     | 0.XX        |

---

## 🎓 Research Collaboration

### For Academic Collaborators

We welcome academic partnerships:

- **Multi-center studies**: Validate across institutions
- **Clinical trials**: Prospective validation with radiologists
- **Comparative studies**: Benchmark against other methods
- **Grant proposals**: Joint research applications

**Contact**: Open an issue labeled `collaboration` or email via GitHub profile.

### Co-authorship Guidelines

For significant research contributions:

1. **Substantial contribution**: New experiments, major code features, clinical validation
2. **Documentation**: Fill out [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) template
3. **Acknowledgment**: Add yourself to `CONTRIBUTORS.md`
4. **Citation**: Update BibTeX with multiple authors if appropriate

---

## 🏆 Recognition

### Contributors List

All contributors will be acknowledged in:

- `CONTRIBUTORS.md` file (to be created)
- GitHub Contributors page
- Research paper acknowledgments (for substantial contributions)

### Contribution Types

We recognize various contribution types using emoji labels:

- 🔬 **Research**: Experimental results, validation studies
- 💻 **Code**: Implementation, bug fixes, optimizations
- 📖 **Documentation**: Guides, tutorials, methodology
- 🎨 **Design**: Visualizations, figures, diagrams
- 🐛 **Bug reports**: Finding and reporting issues
- 💡 **Ideas**: Feature suggestions, architectural improvements

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and respectful environment for all contributors, regardless of:

- Experience level (from students to senior researchers)
- Background (clinical, engineering, computer science)
- Identity (gender, ethnicity, nationality, etc.)

### Expected Behavior

- **Respectful communication**: Constructive feedback, no personal attacks
- **Scientific rigor**: Evidence-based discussions, cite sources
- **Collaborative spirit**: Help others, share knowledge
- **Ethical research**: Follow medical AI ethics guidelines

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Plagiarism or claiming others' work
- Sharing patient data without proper de-identification
- Misrepresenting research results

### Enforcement

Violations will result in warnings, temporary bans, or permanent bans from the project.

**Report issues to**: [Open a confidential issue or contact maintainer]

---

## 🔒 Security & Privacy

### Reporting Security Issues

**Do NOT open public issues for security vulnerabilities.**

Contact maintainer privately via:

- GitHub Security Advisories
- Email (check GitHub profile)

### Data Privacy

When contributing:

- **Never commit** patient data, even if de-identified
- **Never commit** API keys, credentials, or sensitive information
- **Use `.gitignore`** for data folders, checkpoints, logs
- **Follow HIPAA/GDPR** guidelines for any medical data discussions

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the same **MIT License** as the project.

Your contributions are voluntary and you retain copyright, but grant the project maintainers and users the rights specified in the MIT License.

---

## 🙏 Thank You!

Every contribution, no matter how small, helps advance medical AI research and ultimately improves patient care. We appreciate your time and effort!

**Questions?** Open an issue or start a discussion on GitHub.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Maintainer**: techySPHINX  
**Repository**: [github.com/techySPHINX/ResUpNet](https://github.com/techySPHINX/ResUpNet)
