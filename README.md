# ML Model CI/CD Pipeline

This project implements a CI/CD pipeline for a simple CNN model trained on the MNIST dataset. The pipeline includes automated training, testing, and validation steps.

## Project Structure 

project/
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # GitHub Actions workflow configuration
├── train.py # Model architecture and training script
├── test_model.py # Test cases and validation logic
├── pytest.ini # Pytest configuration
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules



## Features

- **Model Architecture**:
  - Simple CNN for MNIST digit classification
  - Less than 25,000 parameters
  - Achieves ≥95% accuracy on test set
  - Input: 28x28 grayscale images
  - Output: 10 classes (digits 0-9)

- **Automated Tests**:
  1. Architecture Validation
     - Verifies input/output dimensions
     - Checks parameter count
  2. Performance Validation
     - Tests model accuracy on MNIST test set
     - Validates accuracy threshold

- **CI/CD Pipeline**:
  - Automated training and testing on push
  - Model artifact storage
  - Test results archival

## Setup and Usage

### Local Development

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Model**:
   ```bash
   python train.py
   ```
   This will:
   - Download MNIST dataset (if needed)
   - Train model for one epoch
   - Save model with timestamp

4. **Run Tests**:
   ```bash
   python test_model.py
   ```
   This will run:
   - Architecture validation
   - Performance validation
   - Display detailed results

### GitHub Integration

1. **Repository Setup**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **GitHub Actions**:
   - Automatically triggers on push
   - Runs training and testing
   - Stores model artifacts
   - Archives test results

## Test Output Format

The tests provide detailed output:

================================================================================
Test 1: Model Architecture Validation
================================================================================
Testing Input/Output Dimensions:
✓ Input shape: (1, 1, 28, 28)
✓ Output shape: (1, 10)
Checking Model Size:
✓ Total parameters: 7,098
✓ Requirement: < 25,000 parameters
================================================================================
Test 2: Model Performance Evaluation
================================================================================
...


## Dependencies

- torch>=2.0.0
- torchvision>=0.15.0
- pytest>=7.0.0
- numpy>=1.21.0

## Model Details

- **Architecture**:
  - 2 Convolutional layers
  - 2 MaxPool layers
  - 2 Fully connected layers
  - ReLU activations
  - Total parameters: ~7,098

- **Training**:
  - Dataset: MNIST
  - Optimizer: Adam
  - Learning rate: 0.001
  - Batch size: 64
  - Epochs: 1

## Notes

- Models are saved with timestamps for versioning
- Tests must pass before deploying
- GitHub Actions runs on CPU
- Test dataset is automatically downloaded

## Troubleshooting

1. **Model Not Found Error**:
   - Ensure you've run `train.py` before testing
   - Check if model file exists in project directory

2. **Dependency Issues**:
   - Verify Python version (3.8 recommended)
   - Update pip: `pip install --upgrade pip`
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **Test Failures**:
   - Check model architecture changes
   - Verify training completed successfully
   - Ensure MNIST dataset downloaded correctly

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is open source and available under the MIT License.

This README.md provides:
Clear project structure
Step-by-step setup instructions
Usage guidelines
Test output examples
Troubleshooting tips
Detailed model information
Contributing guidelines
