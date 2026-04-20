#!/usr/bin/env python3
"""
Demo script for Granite Timeseries TTM-R1 (Tiny Time Mixer)
Issue: tenstorrent/tt-metal#32142

This script demonstrates the PyTorch reference implementation
of the Granite Timeseries TTM-R1 model.
"""

import torch
import sys
import os

# Add reference directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reference'))

from torch_reference import create_ttm_r1, GraniteTTMR1


def demo_basic():
    """Basic demonstration of the model."""
    print("=" * 60)
    print("Granite Timeseries TTM-R1 (Tiny Time Mixer) Demo")
    print("=" * 60)
    print()
    
    # Create model
    print("Creating model...")
    model = create_ttm_r1(num_channels=7)  # ETT dataset (7 features)
    print(f"✓ Model created with {model.count_parameters():,} parameters")
    print()
    
    # Test with different batch sizes
    print("Testing forward pass...")
    for batch_size in [1, 4]:
        x = torch.randn(batch_size, 7, 512)
        with torch.no_grad():
            output = model(x)
        print(f"  Input shape: {x.shape} -> Output shape: {output.shape}")
    
    print()
    print("✓ Demo completed successfully!")
    print()


def demo_multivariate():
    """Demonstrate multivariate forecasting."""
    print("=" * 60)
    print("Multivariate Forecasting Demo")
    print("=" * 60)
    print()
    
    # ETT dataset (7 channels)
    print("ETT Dataset (7 channels):")
    model_ett = create_ttm_r1(num_channels=7)
    x_ett = torch.randn(2, 7, 512)
    with torch.no_grad():
        output_ett = model_ett(x_ett)
    print(f"  Input: {x_ett.shape} -> Output: {output_ett.shape}")
    print(f"  Parameters: {model_ett.count_parameters():,}")
    print()
    
    # Weather dataset (21 channels)
    print("Weather Dataset (21 channels):")
    model_weather = create_ttm_r1(num_channels=21)
    x_weather = torch.randn(2, 21, 512)
    with torch.no_grad():
        output_weather = model_weather(x_weather)
    print(f"  Input: {x_weather.shape} -> Output: {output_weather.shape}")
    print(f"  Parameters: {model_weather.count_parameters():,}")
    print()
    
    # Electricity dataset (321 channels)
    print("Electricity Dataset (321 channels):")
    model_ecl = create_ttm_r1(num_channels=321)
    x_ecl = torch.randn(1, 321, 512)
    with torch.no_grad():
        output_ecl = model_ecl(x_ecl)
    print(f"  Input: {x_ecl.shape} -> Output: {output_ecl.shape}")
    print(f"  Parameters: {model_ecl.count_parameters():,}")
    print()


def demo_model_structure():
    """Show model structure."""
    print("=" * 60)
    print("Model Structure Analysis")
    print("=" * 60)
    print()
    
    model = create_ttm_r1(num_channels=7)
    
    # Count parameters by layer type
    total_params = 0
    layer_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                layer_type = type(module).__name__
                layer_counts[name] = (layer_type, params)
                total_params += params
    
    print(f"Total parameters: {total_params:,}")
    print()
    print("Layer breakdown:")
    for name, (layer_type, params) in sorted(layer_counts.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  {name}: {params:,} ({layer_type})")
    print()


def demo_inference_speed():
    """Measure inference speed."""
    print("=" * 60)
    print("Inference Speed Test")
    print("=" * 60)
    print()
    
    model = create_ttm_r1(num_channels=7)
    model.eval()
    
    # Warmup
    x = torch.randn(1, 7, 512)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure
    import time
    
    batch_size = 1
    num_iterations = 100
    
    x = torch.randn(batch_size, 7, 512)
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    end = time.time()
    
    elapsed = end - start
    throughput = num_iterations / elapsed
    latency_ms = (elapsed / num_iterations) * 1000
    
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.1f} sequences/second")
    print(f"Latency: {latency_ms:.2f}ms per sequence")
    print()
    print(f"Target: 500+ sequences/second (Stage 1)")
    print(f"Target: 2000+ sequences/second (Stage 3)")
    print()


if __name__ == "__main__":
    demo_basic()
    demo_multivariate()
    demo_model_structure()
    demo_inference_speed()
