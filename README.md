# ML Model CI/CD Pipeline

[![ML Pipeline](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/nagalakshmi-nimmagadda/mnist-mlops-pipeline/actions/workflows/ml-pipeline.yml)

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
