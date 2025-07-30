import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from src.models.mlp_interpolant_temporal_nd import MLPTemporalSpectralInterpolation
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP


def setup_models():
    """Set up test models with consistent parameters."""
    device = "cpu"
    Ns = [16, 16]  # Small grid for testing
    bases = ["chebyshev", "chebyshev"]
    domains = [(0, 1), (-1, 1)]
    hidden_layers = (64, 64)
    
    # Create temporal model (time_dim=0)
    temporal_model = MLPTemporalSpectralInterpolation(
        Ns=Ns,
        bases=bases,
        domains=domains,
        device=device,
        time_dim=0,
        hidden_layers=hidden_layers,
        activation=torch.tanh,
    )
    
    # Create regular MLP interpolant for comparison
    mlp_interpolant = MLPSpectralInterpolationND(
        Ns=Ns,
        bases=bases,
        domains=domains,
        device=device,
        hidden_layers=hidden_layers,
        activation=torch.tanh,
    )
    
    return temporal_model, mlp_interpolant


def test_temporal_vs_mlp_interpolant():
    """Test temporal MLP vs MLP interpolant with identical MLP parameters."""
    print("Testing temporal MLP vs MLP interpolant...")
    temporal_model, mlp_interpolant = setup_models()
    
    # Step 1: Copy MLP parameters from temporal to MLP interpolant
    print("1. Copying MLP parameters...")
    with torch.no_grad():
        for temporal_param, mlp_param in zip(temporal_model.mlp.parameters(), mlp_interpolant.mlp.parameters()):
            mlp_param.copy_(temporal_param)
    
    # Step 2: Verify nodal values are identical
    print("2. Testing nodal values...")
    temporal_nodal = temporal_model.nodal_values()
    mlp_nodal = mlp_interpolant.nodal_values()
    
    print(f"   Temporal nodal shape: {temporal_nodal.shape}")
    print(f"   MLP nodal shape: {mlp_nodal.shape}")
    print(f"   Temporal nodal range: [{temporal_nodal.min().item():.6f}, {temporal_nodal.max().item():.6f}]")
    print(f"   MLP nodal range: [{mlp_nodal.min().item():.6f}, {mlp_nodal.max().item():.6f}]")
    
    nodal_diff = torch.abs(temporal_nodal - mlp_nodal)
    print(f"   Max nodal difference: {nodal_diff.max().item():.6e}")
    
    assert torch.allclose(temporal_nodal, mlp_nodal, atol=1e-10), "Nodal values should be identical"
    print("   ✓ Nodal values are identical")
    
    # Step 3: Verify interpolation is identical
    print("3. Testing interpolation...")
    t_points = torch.linspace(0, 1, 5)
    x_points = torch.linspace(-1, 1, 5)
    
    temporal_interp = temporal_model.interpolate([t_points, x_points])
    mlp_interp = mlp_interpolant.interpolate([t_points, x_points])
    
    print(f"   Temporal interp shape: {temporal_interp.shape}")
    print(f"   MLP interp shape: {mlp_interp.shape}")
    print(f"   Temporal interp range: [{temporal_interp.min().item():.6f}, {temporal_interp.max().item():.6f}]")
    print(f"   MLP interp range: [{mlp_interp.min().item():.6f}, {mlp_interp.max().item():.6f}]")
    
    interp_diff = torch.abs(temporal_interp - mlp_interp)
    print(f"   Max interpolation difference: {interp_diff.max().item():.6e}")
    
    assert torch.allclose(temporal_interp, mlp_interp, atol=1e-10), "Interpolation should be identical"
    print("   ✓ Interpolation is identical")
    
    # Step 4: Verify space derivatives are identical
    print("4. Testing space derivatives...")
    temporal_u_x = temporal_model.derivative([t_points, x_points], k=(0, 1))
    mlp_u_x = mlp_interpolant.derivative([t_points, x_points], k=(0, 1))
    
    print(f"   Temporal u_x shape: {temporal_u_x.shape}")
    print(f"   MLP u_x shape: {mlp_u_x.shape}")
    print(f"   Temporal u_x range: [{temporal_u_x.min().item():.6f}, {temporal_u_x.max().item():.6f}]")
    print(f"   MLP u_x range: [{mlp_u_x.min().item():.6f}, {mlp_u_x.max().item():.6f}]")
    
    u_x_diff = torch.abs(temporal_u_x - mlp_u_x)
    print(f"   Max u_x difference: {u_x_diff.max().item():.6e}")
    
    assert torch.allclose(temporal_u_x, mlp_u_x, atol=1e-10), "Space derivatives should be identical"
    print("   ✓ Space derivatives are identical")
    
    # Step 5: Verify time derivatives are different but same shape
    print("5. Testing time derivatives...")
    temporal_u_t = temporal_model.derivative([t_points, x_points], k=(1, 0))
    mlp_u_t = mlp_interpolant.derivative([t_points, x_points], k=(1, 0))
    
    print(f"   Temporal u_t shape: {temporal_u_t.shape}")
    print(f"   MLP u_t shape: {mlp_u_t.shape}")
    print(f"   Temporal u_t range: [{temporal_u_t.min().item():.6f}, {temporal_u_t.max().item():.6f}]")
    print(f"   MLP u_t range: [{mlp_u_t.min().item():.6f}, {mlp_u_t.max().item():.6f}]")
    
    # Check shapes are identical
    assert temporal_u_t.shape == mlp_u_t.shape, f"Time derivative shapes should be identical: {temporal_u_t.shape} vs {mlp_u_t.shape}"
    print("   ✓ Time derivative shapes are identical")
    
    # Check that they are actually different (as expected)
    u_t_diff = torch.abs(temporal_u_t - mlp_u_t)
    print(f"   Max u_t difference: {u_t_diff.max().item():.6e}")
    
    # They should be different because temporal uses MLP autograd, MLP interpolant uses spectral
    assert u_t_diff.max().item() > 1e-6, "Time derivatives should be different (temporal uses MLP autograd, MLP interpolant uses spectral)"
    print("   ✓ Time derivatives are different (as expected)")
    
    print("✓ All tests passed!")


def test_mlp_interpolant_autograd_mode():
    """Test that MLP interpolant with autograd gives same results as temporal model for time derivatives."""
    print("Testing MLP interpolant in autograd mode...")
    temporal_model, mlp_interpolant = setup_models()
    
    # Copy MLP parameters
    with torch.no_grad():
        for temporal_param, mlp_param in zip(temporal_model.mlp.parameters(), mlp_interpolant.mlp.parameters()):
            mlp_param.copy_(temporal_param)
    
    # Set MLP interpolant to use autograd for derivatives
    mlp_interpolant.use_mlp_for_derivatives = True
    
    # Create test points
    t_points = torch.linspace(0, 1, 5)
    x_points = torch.linspace(-1, 1, 5)
    
    # Test time derivatives - these should now be identical
    temporal_u_t = temporal_model.derivative([t_points, x_points], k=(1, 0))
    mlp_u_t = mlp_interpolant.derivative([t_points, x_points], k=(1, 0))
    
    print(f"   Temporal u_t range: [{temporal_u_t.min().item():.6f}, {temporal_u_t.max().item():.6f}]")
    print(f"   MLP u_t range: [{mlp_u_t.min().item():.6f}, {mlp_u_t.max().item():.6f}]")
    
    u_t_diff = torch.abs(temporal_u_t - mlp_u_t)
    print(f"   Max u_t difference: {u_t_diff.max().item():.6e}")
    
    # They should now be identical because both use MLP autograd
    assert torch.allclose(temporal_u_t, mlp_u_t, atol=1e-6), "Time derivatives should be identical when both use MLP autograd"
    print("   ✓ Time derivatives are identical (both using MLP autograd)")
    
    print("✓ MLP interpolant autograd mode test passed!")


def run_tests():
    """Run the focused comparison tests."""
    print("=" * 60)
    print("Running temporal MLP vs MLP interpolant tests...")
    print("=" * 60)
    
    tests = [
        test_temporal_vs_mlp_interpolant,
        test_mlp_interpolant_autograd_mode,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 