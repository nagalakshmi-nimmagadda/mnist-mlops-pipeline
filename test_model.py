import torch
import torch.nn as nn
from train import SimpleCNN
from torchvision import datasets, transforms
import glob
import pytest

def get_latest_model():
    model_files = glob.glob('model_*.pth')
    return max(model_files) if model_files else None

@pytest.mark.architecture
def test_model_architecture():
    """Validates CNN Architecture Requirements"""
    print("\n" + "="*80)
    print("Test 1: Model Architecture Validation")
    print("="*80)
    
    model = SimpleCNN()
    
    # 1. Input Shape Test
    print("\n1. Testing Input/Output Dimensions:")
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"  ✓ Input shape: (1, 1, 28, 28)")
    print(f"  ✓ Output shape: {tuple(output.shape)}")
    assert output.shape == (1, 10), "Model output should be 10 classes"
    
    # 2. Parameter Count Test
    print("\n2. Checking Model Size:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Requirement: < 25,000 parameters")
    assert total_params < 25000, f"Model has {total_params:,} parameters, should be less than 25,000"

@pytest.mark.performance
def test_model_accuracy():
    """Validates Model Performance on MNIST Test Dataset"""
    print("\n" + "="*80)
    print("Test 2: Model Performance Evaluation")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # 1. Model Loading
    print("\n1. Loading Trained Model:")
    model_path = get_latest_model()
    assert model_path is not None, "No trained model found"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"  ✓ Successfully loaded model: {model_path}")
    
    # 2. Dataset Preparation
    print("\n2. Preparing MNIST Test Dataset:")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    print(f"  ✓ Test dataset size: {len(test_dataset)} images")
    
    # 3. Accuracy Evaluation
    print("\n3. Evaluating Model Accuracy:")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"  ✓ Test Accuracy: {accuracy:.2f}%")
    print(f"  ✓ Requirement: ≥ 95.00%")
    assert accuracy >= 95, f"Model accuracy on test set is {accuracy:.2f}%, should be at least 95%"

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "architecture: mark test as architecture validation"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance validation"
    )

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--no-header",
        "-s"
    ]) 