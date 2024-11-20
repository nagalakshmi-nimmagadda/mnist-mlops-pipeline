# ML Model CI/CD Pipeline

[![ML Pipeline](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline/actions/workflows/ml-pipeline.yml)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline/actions)

[![Model Size](https://img.shields.io/badge/Model%20Size-<25K%20params-success.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline)
[![Accuracy](https://img.shields.io/badge/Accuracy-≥95%25-brightgreen.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey.svg)](http://yann.lecun.com/exdb/mnist/)
[![Inference Time](https://img.shields.io/badge/Inference-<1s-blue.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline)

## 🖼️ Augmentation Examples
![Augmentation Samples](augmentation_samples.png)

A lightweight CNN-based MNIST classifier with complete CI/CD pipeline implementation. Features automated training, testing, and validation using GitHub Actions. The model achieves >95% accuracy with <25K parameters in single epoch training.

## 📋 Project Structure 

```
mnist-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml    # GitHub Actions workflow
├── train.py                   # Model and training logic
├── test_model.py             # Test suite
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Dependencies
└── .gitignore               # Git ignore rules
```

## ✨ Features

### 🧠 Model Architecture
- Simple CNN for MNIST digit classification
- Less than 25,000 parameters
- Achieves ≥95% accuracy on test set
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

### 🔄 Automated Tests
1. Architecture Validation
   - Verifies input/output dimensions
   - Checks parameter count
2. Performance Validation
   - Tests model accuracy on MNIST test set
   - Validates accuracy threshold

### 🚀 CI/CD Pipeline
- Automated training and testing on push
- Model artifact storage
- Test results archival

## 🛠️ Setup and Usage

### Local Development

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Train Model**
```bash
python train.py
```
This will:
- Download MNIST dataset (if needed)
- Train model for one epoch
- Save model with timestamp

4. **Run Tests**
```bash
python test_model.py
```

### GitHub Integration

1. **Repository Setup**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

2. **GitHub Actions** will automatically:
- Trigger on push
- Run training and testing
- Store model artifacts
- Archive test results

## 📊 Test Output Format

```
================================================================================
Test 1: Model Architecture Validation
================================================================================
1. Testing Input/Output Dimensions:
  ✓ Input shape: (1, 1, 28, 28)
  ✓ Output shape: (1, 10)

2. Checking Model Size:
  ✓ Total parameters: 7,098
  ✓ Requirement: < 25,000 parameters
```

## 📦 Dependencies

- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `pytest>=7.0.0`
- `numpy>=1.21.0`

## 🔍 Model Details

### Architecture
- 2 Convolutional layers
- 2 MaxPool layers
- 2 Fully connected layers
- ReLU activations
- Total parameters: ~7,098

### Training
- Dataset: MNIST
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 1

## 📝 Notes

- Models are saved with timestamps for versioning
- Tests must pass before deploying
- GitHub Actions runs on CPU
- Test dataset is automatically downloaded

## ❗ Troubleshooting

### Model Not Found Error
- Ensure you've run `train.py` before testing
- Check if model file exists in project directory

### Dependency Issues
- Verify Python version (3.8 recommended)
- Update pip: `pip install --upgrade pip`
- Reinstall dependencies: `pip install -r requirements.txt`

### Test Failures
- Check model architecture changes
- Verify training completed successfully
- Ensure MNIST dataset downloaded correctly

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
Made with ❤️ using PyTorch and GitHub Actions
