#!/usr/bin/env python3
"""
Test script to verify PirateNet works with PDE experiments
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.models.piratenet import PirateNet
from src.experiments.pdes.simple.convection import Convection

def test_piratenet_pde_integration():
    """Test that PirateNet can be used in PDE experiments"""
    print("=== Testing PirateNet PDE Integration ===")
    
    # Create a simple convection PDE
    pde = Convection(c=2, device="cpu")
    
    # Create PirateNet model
    model = PirateNet(
        n_dim=2,
        n_layers=2,
        hidden_dim=32,
        activation="tanh",
        device="cpu"
    )
    
    print(f"PDE: {pde}")
    print(f"Model: {model}")
    
    # Test that the model can handle PDE inputs
    batch_size = 10
    x = torch.randn(batch_size, 2, requires_grad=True)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test that we can compute gradients
    loss = torch.nn.functional.mse_loss(output.squeeze(), torch.zeros(batch_size))
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradients computed successfully")
    
    # Test grid evaluation
    t_nodes = torch.linspace(0, 1, 5)
    x_nodes = torch.linspace(0, 2*torch.pi, 6)
    
    grid_output = model([t_nodes, x_nodes])
    print(f"Grid output shape: {grid_output.shape}")
    
    print("✓ PirateNet PDE integration test passed\n")
    return True

def test_piratenet_training_ready():
    """Test that PirateNet is ready for training"""
    print("=== Testing PirateNet Training Readiness ===")
    
    # Create model
    model = PirateNet(
        n_dim=2,
        n_layers=3,
        hidden_dim=64,
        activation="tanh",
        device="cpu"
    )
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Test optimizer compatibility
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Test training step
    x = torch.randn(100, 2, requires_grad=True)
    target = torch.sin(x[:, 0]) * torch.cos(x[:, 1])
    
    optimizer.zero_grad()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output.squeeze(), target)
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed successfully")
    print(f"Loss after one step: {loss.item():.6f}")
    
    print("✓ PirateNet training readiness test passed\n")
    return True

def main():
    """Run all tests"""
    print("Starting PirateNet PDE Integration Tests\n")
    
    tests = [
        test_piratenet_pde_integration,
        test_piratenet_training_ready,
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
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PirateNet is ready for PDE training.")
        print("\nYou can now use PirateNet in PDE experiments with:")
        print("python -m src.experiments.pdes.simple.convection --model piratenet --method ssbroyden --n_layers 3 --hidden_dim 128 --activation tanh")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main() 