# Configuration for Granite Timeseries TTM-R1 TTNN Implementation
# Issue: tenstorrent/tt-metal#32142

"""
Configuration module for Granite TTM-R1 TTNN implementation.
Contains model configuration, preprocessing utilities, and weight handling.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class GraniteTTMConfig:
    """Configuration for Granite Timeseries TTM-R1 model."""
    
    # Model architecture
    context_length: int = 512       # Input context length
    forecast_length: int = 96       # Output forecast length
    num_channels: int = 1           # Number of input channels (variables)
    d_model: int = 64               # Model dimension
    patch_size: int = 16            # Patch size for adaptive patching
    num_layers: int = 3             # Number of mixer blocks
    dropout: float = 0.1            # Dropout rate
    
    # Hardware configuration
    batch_size: int = 1            # Batch size for inference
    dtype: str = "bfloat16"        # Data type
    
    @property
    def num_patches(self) -> int:
        """Number of patches in the input."""
        return self.context_length // self.patch_size
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.context_length % self.patch_size == 0, \
            f"context_length ({self.context_length}) must be divisible by patch_size ({self.patch_size})"
        assert self.d_model % 32 == 0, \
            f"d_model ({self.d_model}) should be a multiple of 32 for optimal TTNN performance"


def create_model_config(
    context_length: int = 512,
    forecast_length: int = 96,
    num_channels: int = 1,
    d_model: int = 64,
    patch_size: int = 16,
    num_layers: int = 3,
    dropout: float = 0.1,
    batch_size: int = 1,
) -> GraniteTTMConfig:
    """
    Create a Granite TTM configuration.
    
    Args:
        context_length: Input context length (default: 512)
        forecast_length: Output forecast length (default: 96)
        num_channels: Number of input channels (default: 1)
        d_model: Model dimension (default: 64)
        patch_size: Patch size (default: 16)
        num_layers: Number of mixer layers (default: 3)
        dropout: Dropout rate (default: 0.1)
        batch_size: Batch size (default: 1)
        
    Returns:
        GraniteTTMConfig instance
    """
    return GraniteTTMConfig(
        context_length=context_length,
        forecast_length=forecast_length,
        num_channels=num_channels,
        d_model=d_model,
        patch_size=patch_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
    )


def get_default_config() -> GraniteTTMConfig:
    """
    Get the default Granite TTM-R1 configuration.
    
    Returns:
        Default GraniteTTMConfig
    """
    return GraniteTTMConfig(
        context_length=512,
        forecast_length=96,
        num_channels=1,
        d_model=64,
        patch_size=16,
        num_layers=3,
        dropout=0.1,
        batch_size=1,
    )


def get_ett_config() -> GraniteTTMConfig:
    """
    Get configuration optimized for ETT datasets.
    
    Returns:
        ETT-optimized GraniteTTMConfig
    """
    return GraniteTTMConfig(
        context_length=512,
        forecast_length=96,
        num_channels=7,  # ETT has 7 features
        d_model=64,
        patch_size=16,
        num_layers=3,
        dropout=0.1,
        batch_size=1,
    )


def get_weather_config() -> GraniteTTMConfig:
    """
    Get configuration for Weather dataset.
    
    Returns:
        Weather-optimized GraniteTTMConfig
    """
    return GraniteTTMConfig(
        context_length=512,
        forecast_length=96,
        num_channels=21,  # Weather has 21 features
        d_model=64,
        patch_size=16,
        num_layers=3,
        dropout=0.1,
        batch_size=1,
    )


def get_electricity_config() -> GraniteTTMConfig:
    """
    Get configuration for Electricity dataset.
    
    Returns:
        Electricity-optimized GraniteTTMConfig
    """
    return GraniteTTMConfig(
        context_length=512,
        forecast_length=96,
        num_channels=321,  # ECL has 321 clients
        d_model=64,
        patch_size=16,
        num_layers=3,
        dropout=0.1,
        batch_size=1,
    )
