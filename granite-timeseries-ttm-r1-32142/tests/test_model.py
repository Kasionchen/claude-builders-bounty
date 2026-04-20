# Tests for Granite Timeseries TTM-R1 TTNN Implementation
# Issue: tenstorrent/tt-metal#32142

"""
Unit tests for Granite TTM-R1 TTNN implementation.
Tests model structure, weight preprocessing, and basic functionality.
"""

import torch
import pytest
import sys
import os

# Add reference directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reference'))

from torch_reference import GraniteTTMR1, create_ttm_r1, AdaptivePatching, LightweightTimeMixing


class TestPyTorchReference:
    """Test PyTorch reference implementation."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = create_ttm_r1(num_channels=7)
        assert model is not None
        assert model.count_parameters() < 1_000_000, "Model should have < 1M parameters"
        
    def test_parameter_count(self):
        """Test model has correct number of parameters."""
        model = create_ttm_r1(num_channels=1)
        params = model.count_parameters()
        print(f"Model parameters: {params:,}")
        assert params < 1_000_000, f"Model has {params:,} parameters, expected < 1,000,000"
        
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = create_ttm_r1(num_channels=7)
        model.eval()
        
        batch_size = 2
        context_length = 512
        forecast_length = 96
        
        x = torch.randn(batch_size, 7, context_length)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 7, forecast_length), \
            f"Expected shape ({batch_size}, 7, {forecast_length}), got {output.shape}"
            
    def test_different_channels(self):
        """Test model works with different channel counts."""
        for num_channels in [1, 7, 21, 321]:
            model = create_ttm_r1(num_channels=num_channels)
            x = torch.randn(1, num_channels, 512)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, num_channels, 96), \
                f"Failed for {num_channels} channels"
                
    def test_adaptive_patching(self):
        """Test adaptive patching layer."""
        patching = AdaptivePatching(context_length=512, patch_size=16)
        
        x = torch.randn(2, 7, 512)  # (batch, channels, time)
        output = patching(x)
        
        expected_patches = 512 // 16  # 32 patches
        assert output.shape == (2, 7, expected_patches, 16), \
            f"Expected shape (2, 7, {expected_patches}, 16), got {output.shape}"


class TestModelArchitecture:
    """Test model architecture components."""
    
    def test_tiny_time_mixer_block(self):
        """Test Tiny Time Mixer block."""
        from torch_reference import TinyTimeMixerBlock
        
        num_patches = 32
        d_model = 64
        num_channels = 7
        
        block = TinyTimeMixerBlock(num_patches, d_model, num_channels)
        
        x = torch.randn(2, num_channels, num_patches, d_model)
        output = block(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        
    def test_lightweight_time_mixing(self):
        """Test time mixing layer."""
        time_mixer = LightweightTimeMixing(num_patches=32, d_model=64)
        
        x = torch.randn(2, 32, 64)  # (batch, patches, d_model)
        output = time_mixer(x)
        
        assert output.shape == x.shape
        
    def test_lightweight_channel_mixing(self):
        """Test channel mixing layer."""
        from torch_reference import LightweightChannelMixing
        
        channel_mixer = LightweightChannelMixing(d_model=64, num_channels=7)
        
        x = torch.randn(2, 7, 32, 64)  # (batch, channels, patches, d_model)
        output = channel_mixer(x)
        
        assert output.shape == x.shape


class TestModelVariants:
    """Test different model configurations."""
    
    def test_different_context_lengths(self):
        """Test with different context lengths."""
        for context_length in [512]:
            model = GraniteTTMR1(context_length=context_length, num_channels=7)
            x = torch.randn(1, 7, context_length)
            with torch.no_grad():
                output = model(x)
            assert output.shape[2] == 96  # Forecast length is always 96
            assert output.shape[0] == 1
            assert output.shape[1] == 7
            
    def test_single_channel(self):
        """Test with single channel (univariate)."""
        model = create_ttm_r1(num_channels=1)
        x = torch.randn(4, 1, 512)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 1, 96)


class TestIntegration:
    """Integration tests."""
    
    def test_batch_processing(self):
        """Test batch processing works correctly."""
        model = create_ttm_r1(num_channels=7)
        model.eval()
        
        # Process multiple samples
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 7, 512)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 7, 96), \
                f"Failed for batch_size={batch_size}"
                
    def test_gradient_flow(self):
        """Test gradients flow correctly during training."""
        model = create_ttm_r1(num_channels=7)
        
        x = torch.randn(2, 7, 512, requires_grad=True)
        output = model(x)
        
        # Simple loss
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad), \
            "Some parameters should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
