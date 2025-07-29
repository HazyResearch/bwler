#!/usr/bin/env python3
"""
Test script to validate PirateNet compatibility with MLP interface
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union

# Import the models
from src.models.mlp import MLP
from src.models.piratenet import PirateNet

def test_basic_initialization():
    """Test that both models can be initialized with similar parameters"""
    print("=== Testing Basic Initialization ===")
    
    # Test MLP initialization
    mlp = MLP(n_dim=2, n_layers=3, hidden_dim=64, activation=torch.tanh, device="cpu")
    print(f"MLP initialized successfully: {mlp}")
    
    # Test PirateNet initialization with same parameters
    piratenet = PirateNet(n_dim=2, n_layers=3, hidden_dim=64, activation="tanh", device="cpu")
    print(f"PirateNet initialized successfully: {piratenet}")
    
    print("✓ Basic initialization test passed\n")
    return True

def test_input_handling():
    """Test that both models handle the same input types correctly"""
    print("=== Testing Input Handling ===")
    
    # Create test data
    batch_size = 10
    n_dim = 2
    
    # Batch input
    x_batch = torch.randn(batch_size, n_dim)
    print(f"Batch input shape: {x_batch.shape}")
    
    # Grid input
    x_grid = [torch.linspace(0, 1, 5), torch.linspace(0, 1, 6)]
    print(f"Grid input shapes: {[xi.shape for xi in x_grid]}")
    
    # Initialize models
    mlp = MLP(n_dim=n_dim, n_layers=2, hidden_dim=32, activation=torch.tanh, device="cpu")
    piratenet = PirateNet(n_dim=n_dim, n_layers=2, hidden_dim=32, activation="tanh", device="cpu")
    
    # Test batch forward pass
    try:
        mlp_output = mlp(x_batch)
        piratenet_output = piratenet(x_batch)
        print(f"MLP batch output shape: {mlp_output.shape}")
        print(f"PirateNet batch output shape: {piratenet_output.shape}")
        assert mlp_output.shape == piratenet_output.shape, f"Output shapes don't match: {mlp_output.shape} vs {piratenet_output.shape}"
        print("✓ Batch forward pass test passed")
    except Exception as e:
        print(f"✗ Batch forward pass test failed: {e}")
        return False
    
    # Test grid forward pass
    try:
        mlp_grid_output = mlp(x_grid)
        piratenet_grid_output = piratenet(x_grid)
        print(f"MLP grid output shape: {mlp_grid_output.shape}")
        print(f"PirateNet grid output shape: {piratenet_grid_output.shape}")
        assert mlp_grid_output.shape == piratenet_grid_output.shape, f"Grid output shapes don't match: {mlp_grid_output.shape} vs {piratenet_grid_output.shape}"
        print("✓ Grid forward pass test passed")
    except Exception as e:
        print(f"✗ Grid forward pass test failed: {e}")
        return False
    
    print("✓ Input handling test passed\n")
    return True

def test_method_compatibility():
    """Test that both models have the same interface methods"""
    print("=== Testing Method Compatibility ===")
    
    mlp = MLP(n_dim=2, n_layers=2, hidden_dim=32, activation=torch.tanh, device="cpu")
    piratenet = PirateNet(n_dim=2, n_layers=2, hidden_dim=32, activation="tanh", device="cpu")
    
    # Check required methods exist
    required_methods = ['forward', 'make_grid', 'forward_grid', 'forward_batch', 'interpolate']
    
    for method in required_methods:
        assert hasattr(mlp, method), f"MLP missing method: {method}"
        assert hasattr(piratenet, method), f"PirateNet missing method: {method}"
        print(f"✓ Both models have method: {method}")
    
    print("✓ Method compatibility test passed\n")
    return True

def test_output_consistency():
    """Test that outputs are consistent and have correct shapes"""
    print("=== Testing Output Consistency ===")
    
    # Test with different input sizes
    test_cases = [
        (1, 2),  # (batch_size, n_dim)
        (5, 2),
        (10, 2),
        (1, 3),  # 3D input
    ]
    
    for batch_size, n_dim in test_cases:
        print(f"Testing with batch_size={batch_size}, n_dim={n_dim}")
        
        mlp = MLP(n_dim=n_dim, n_layers=2, hidden_dim=32, activation=torch.tanh, device="cpu")
        piratenet = PirateNet(n_dim=n_dim, n_layers=2, hidden_dim=32, activation="tanh", device="cpu")
        
        x = torch.randn(batch_size, n_dim)
        
        mlp_output = mlp(x)
        piratenet_output = piratenet(x)
        
        print(f"  MLP output shape: {mlp_output.shape}")
        print(f"  PirateNet output shape: {piratenet_output.shape}")
        
        # Check shapes match
        assert mlp_output.shape == piratenet_output.shape, f"Shape mismatch: {mlp_output.shape} vs {piratenet_output.shape}"
        
        # Check output is scalar (last dimension should be 1)
        assert mlp_output.shape[-1] == 1, f"MLP output should have last dim=1, got {mlp_output.shape[-1]}"
        assert piratenet_output.shape[-1] == 1, f"PirateNet output should have last dim=1, got {piratenet_output.shape[-1]}"
        
        print(f"  ✓ Test case passed")
    
    print("✓ Output consistency test passed\n")
    return True

def test_grid_interpolation():
    """Test grid interpolation functionality"""
    print("=== Testing Grid Interpolation ===")
    
    n_dim = 2
    grid_sizes = [5, 6]
    
    mlp = MLP(n_dim=n_dim, n_layers=2, hidden_dim=32, activation=torch.tanh, device="cpu")
    piratenet = PirateNet(n_dim=n_dim, n_layers=2, hidden_dim=32, activation="tanh", device="cpu")
    
    # Create grid input
    x_grid = [torch.linspace(0, 1, grid_sizes[0]), torch.linspace(0, 1, grid_sizes[1])]
    
    # Test make_grid method
    mlp_grid = mlp.make_grid(x_grid)
    piratenet_grid = piratenet.make_grid(x_grid)
    
    print(f"MLP grid shape: {mlp_grid.shape}")
    print(f"PirateNet grid shape: {piratenet_grid.shape}")
    assert mlp_grid.shape == piratenet_grid.shape, f"Grid shapes don't match: {mlp_grid.shape} vs {piratenet_grid.shape}"
    
    # Test forward_grid method
    mlp_output = mlp.forward_grid(mlp_grid)
    piratenet_output = piratenet.forward_grid(piratenet_grid)
    
    print(f"MLP grid output shape: {mlp_output.shape}")
    print(f"PirateNet grid output shape: {piratenet_output.shape}")
    assert mlp_output.shape == piratenet_output.shape, f"Grid output shapes don't match: {mlp_output.shape} vs {piratenet_output.shape}"
    
    # Test interpolate method
    mlp_interp = mlp.interpolate(x_grid)
    piratenet_interp = piratenet.interpolate(x_grid)
    
    print(f"MLP interpolate output shape: {mlp_interp.shape}")
    print(f"PirateNet interpolate output shape: {piratenet_interp.shape}")
    assert mlp_interp.shape == piratenet_interp.shape, f"Interpolate output shapes don't match: {mlp_interp.shape} vs {piratenet_interp.shape}"
    
    print("✓ Grid interpolation test passed\n")
    return True

def test_activation_compatibility():
    """Test that activation functions work correctly"""
    print("=== Testing Activation Compatibility ===")
    
    activations = ["tanh", "relu", "gelu", "silu"]
    
    for activation in activations:
        print(f"Testing activation: {activation}")
        
        # For MLP, we need to convert string to function
        if activation == "tanh":
            mlp_activation = torch.tanh
        elif activation == "relu":
            mlp_activation = torch.relu
        elif activation == "gelu":
            mlp_activation = torch.nn.functional.gelu
        elif activation == "silu":
            mlp_activation = torch.nn.functional.silu
        else:
            continue
        
        mlp = MLP(n_dim=2, n_layers=2, hidden_dim=32, activation=mlp_activation, device="cpu")
        piratenet = PirateNet(n_dim=2, n_layers=2, hidden_dim=32, activation=activation, device="cpu")
        
        x = torch.randn(5, 2)
        
        mlp_output = mlp(x)
        piratenet_output = piratenet(x)
        
        print(f"  MLP output range: [{mlp_output.min().item():.3f}, {mlp_output.max().item():.3f}]")
        print(f"  PirateNet output range: [{piratenet_output.min().item():.3f}, {piratenet_output.max().item():.3f}]")
        
        # Check that outputs are finite
        assert torch.isfinite(mlp_output).all(), "MLP output contains non-finite values"
        assert torch.isfinite(piratenet_output).all(), "PirateNet output contains non-finite values"
        
        print(f"  ✓ {activation} activation test passed")
    
    print("✓ Activation compatibility test passed\n")
    return True

def test_device_compatibility():
    """Test that both models work on CPU and GPU if available"""
    print("=== Testing Device Compatibility ===")
    
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    
    for device in devices:
        print(f"Testing on device: {device}")
        
        mlp = MLP(n_dim=2, n_layers=2, hidden_dim=32, activation=torch.tanh, device=device)
        piratenet = PirateNet(n_dim=2, n_layers=2, hidden_dim=32, activation="tanh", device=device)
        
        x = torch.randn(5, 2, device=device)
        
        mlp_output = mlp(x)
        piratenet_output = piratenet(x)
        
        print(f"  MLP output device: {mlp_output.device}")
        print(f"  PirateNet output device: {piratenet_output.device}")
        
        # Check that outputs are on the same device as the input
        assert mlp_output.device == x.device, f"MLP output device {mlp_output.device} doesn't match input device {x.device}"
        assert piratenet_output.device == x.device, f"PirateNet output device {piratenet_output.device} doesn't match input device {x.device}"
        
        print(f"  ✓ {device} device test passed")
    
    print("✓ Device compatibility test passed\n")
    return True

def test_pde_integration():
    """Test that PirateNet can be used in PDE experiments"""
    print("=== Testing PDE Integration ===")
    
    # Simulate a simple PDE-like scenario
    n_dim = 2
    batch_size = 100
    
    # Create a simple function to approximate: f(x, y) = sin(x) * cos(y)
    x = torch.randn(batch_size, n_dim)
    target = torch.sin(x[:, 0]) * torch.cos(x[:, 1])
    
    mlp = MLP(n_dim=n_dim, n_layers=3, hidden_dim=64, activation=torch.tanh, device="cpu")
    piratenet = PirateNet(n_dim=n_dim, n_layers=3, hidden_dim=64, activation="tanh", device="cpu")
    
    # Test forward pass
    mlp_output = mlp(x)
    piratenet_output = piratenet(x)
    
    print(f"MLP output shape: {mlp_output.shape}")
    print(f"PirateNet output shape: {piratenet_output.shape}")
    print(f"Target shape: {target.shape}")
    
    # Check that outputs have correct shape for loss computation
    assert mlp_output.shape == (batch_size, 1), f"MLP output shape incorrect: {mlp_output.shape}"
    assert piratenet_output.shape == (batch_size, 1), f"PirateNet output shape incorrect: {piratenet_output.shape}"
    
    # Test that we can compute MSE loss
    mlp_loss = torch.nn.functional.mse_loss(mlp_output.squeeze(), target)
    piratenet_loss = torch.nn.functional.mse_loss(piratenet_output.squeeze(), target)
    
    print(f"MLP MSE loss: {mlp_loss.item():.6f}")
    print(f"PirateNet MSE loss: {piratenet_loss.item():.6f}")
    
    # Check that losses are finite
    assert torch.isfinite(mlp_loss), "MLP loss is not finite"
    assert torch.isfinite(piratenet_loss), "PirateNet loss is not finite"
    
    print("✓ PDE integration test passed\n")
    return True

def main():
    """Run all compatibility tests"""
    print("Starting PirateNet-MLP Compatibility Tests\n")
    
    tests = [
        test_basic_initialization,
        test_input_handling,
        test_method_compatibility,
        test_output_consistency,
        test_grid_interpolation,
        test_activation_compatibility,
        test_device_compatibility,
        test_pde_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test {test.__name__} failed")
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PirateNet is compatible with MLP interface.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main() 