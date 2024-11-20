import torch
import torch.nn as nn
from train import SimpleCNN
from torchvision import datasets, transforms
import glob
import pytest
import time

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

@pytest.mark.robustness
def test_model_robustness():
    """Tests model's robustness to input variations"""
    print("\n" + "="*80)
    print("Test 3: Model Robustness Evaluation")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Test with different input perturbations
    test_input = torch.randn(1, 1, 28, 28).to(device)
    
    # 1. Noise resistance
    print("\n1. Testing Noise Resistance:")
    noisy_input = test_input + 0.1 * torch.randn_like(test_input)
    original_output = model(test_input)
    noisy_output = model(noisy_input)
    output_diff = torch.norm(original_output - noisy_output)
    print(f"  ✓ Output difference with noise: {output_diff:.4f}")
    assert output_diff < 1.0, "Model is too sensitive to input noise"

    # 2. Scale invariance
    print("\n2. Testing Scale Invariance:")
    scaled_input = 0.9 * test_input
    scaled_output = model(scaled_input)
    scale_diff = torch.norm(original_output - scaled_output)
    print(f"  ✓ Output difference with scaling: {scale_diff:.4f}")
    assert scale_diff < 1.0, "Model is too sensitive to input scaling"

@pytest.mark.memory
def test_model_memory_usage():
    """Tests model's memory efficiency"""
    print("\n" + "="*80)
    print("Test 4: Memory Usage Evaluation")
    print("="*80)
    
    model = SimpleCNN()
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"  ✓ Model Size: {size_mb:.2f} MB")
    assert size_mb < 1.0, f"Model size ({size_mb:.2f} MB) exceeds 1 MB limit"

@pytest.mark.inference
def test_model_inference_time():
    """Tests model's inference speed"""
    print("\n" + "="*80)
    print("Test 5: Inference Speed Evaluation")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Warm-up
    for _ in range(10):
        _ = model(torch.randn(1, 1, 28, 28).to(device))
    
    # Test batch inference time
    batch_sizes = [1, 32, 64]
    for batch_size in batch_sizes:
        input_batch = torch.randn(batch_size, 1, 28, 28).to(device)
        
        # Use time.perf_counter for both CPU and GPU
        times = []
        
        # Run multiple iterations for more stable timing
        n_iterations = 100
        with torch.no_grad():
            for _ in range(n_iterations):
                start_time = time.perf_counter()
                _ = model(input_batch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Wait for GPU
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate average time
        avg_time = sum(times) / len(times)
        print(f"  ✓ Batch size {batch_size}: {avg_time:.2f} ms")
        
        # Adjust threshold based on device
        threshold = 100 if device.type == 'cuda' else 1000  # Higher threshold for CPU
        assert avg_time < threshold, f"Inference too slow for batch size {batch_size}"

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "architecture: mark test as architecture validation"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance validation"
    )
    config.addinivalue_line("markers", "robustness: mark test as robustness validation")
    config.addinivalue_line("markers", "memory: mark test as memory usage validation")
    config.addinivalue_line("markers", "inference: mark test as inference speed validation")

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--no-header",
        "-s"
    ]) 