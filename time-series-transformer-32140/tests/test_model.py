# Test module for Time Series Transformer TTNN Implementation
# Issue: tenstorrent/tt-metal#32140

"""Tests for Time Series Transformer TTNN implementation."""

import torch
import pytest


def test_torch_model_creation():
    """Test that the PyTorch reference model can be created."""
    from reference.torch_reference import create_model, TimeSeriesTransformerForPrediction
    
    model = create_model(
        context_length=512,
        prediction_length=96,
        input_size=1,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
    )
    
    assert model is not None
    assert isinstance(model, TimeSeriesTransformerForPrediction)


def test_torch_model_forward():
    """Test that the PyTorch reference model forward pass works."""
    from reference.torch_reference import create_model
    
    model = create_model(
        context_length=512,
        prediction_length=96,
        input_size=1,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
    )
    model.eval()
    
    # Create sample inputs
    batch_size = 2
    past_values = torch.randn(batch_size, 512, 1)
    past_time_features = torch.randn(batch_size, 512, 5)
    future_time_features = torch.randn(batch_size, 96, 5)
    
    # Forward pass
    with torch.no_grad():
        dist_params = model(
            past_values,
            past_time_features,
            future_time_features,
        )
    
    assert "loc" in dist_params
    assert "scale" in dist_params
    assert dist_params["loc"].shape == (batch_size, 96, 1)
    assert dist_params["scale"].shape == (batch_size, 96, 1)


def test_config_defaults():
    """Test that the TTNN config has correct defaults."""
    from tt.config import get_default_config, TimeSeriesTransformerConfig
    
    config = get_default_config()
    
    assert isinstance(config, TimeSeriesTransformerConfig)
    assert config.context_length == 512
    assert config.prediction_length == 96
    assert config.input_size == 1
    assert config.d_model == 128


def test_tourism_config():
    """Test tourism dataset config."""
    from tt.config import get_tourism_config
    
    config = get_tourism_config()
    
    assert config.prediction_length == 24  # Tourism uses shorter predictions
    assert config.d_model == 64


def test_ett_config():
    """Test ETT dataset config."""
    from tt.config import get_ett_config
    
    config = get_ett_config()
    
    assert config.input_size == 7  # ETT has 7 features
    assert config.prediction_length == 96


if __name__ == "__main__":
    test_torch_model_creation()
    test_torch_model_forward()
    test_config_defaults()
    test_tourism_config()
    test_ett_config()
    print("All tests passed!")
