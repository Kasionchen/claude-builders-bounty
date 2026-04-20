# Configuration for Time Series Transformer TTNN Implementation
# Issue: tenstorrent/tt-metal#32140
# Bounty: $1500

"""
Configuration module for Time Series Transformer TTNN implementation.
Implements a vanilla encoder-decoder Transformer for probabilistic time-series forecasting.
Based on HuggingFace Transformers Time Series Transformer architecture.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch


@dataclass
class TimeSeriesTransformerConfig:
    """Configuration for Time Series Transformer model."""
    
    # Model architecture
    context_length: int = 512       # Input context length (past values)
    prediction_length: int = 96     # Output prediction length (future values)
    input_size: int = 1             # Number of target variables
    num_time_features: int = 5      # Number of time features (hour, day, weekday, month, day_of_year)
    
    # Transformer architecture
    d_model: int = 128             # Model dimension
    encoder_layers: int = 2         # Number of encoder layers
    decoder_layers: int = 2         # Number of decoder layers
    encoder_attention_heads: int = 4  # Number of attention heads
    decoder_attention_heads: int = 4  # Number of attention heads
    encoder_ffn_dim: int = 512      # Encoder feed-forward dimension
    decoder_ffn_dim: int = 512      # Decoder feed-forward dimension
    dropout: float = 0.1            # Dropout rate
    
    # Distribution parameters
    distribution_output: str = "student_t"  # Distribution type: student_t, normal, negative_binomial
    
    # Feature dimensions
    num_static_categorical_features: int = 0
    num_static_real_features: int = 0
    cardinality: List[int] = []     # List of cardinalities for categorical features
    embedding_dimension: int = 16    # Dimension for categorical embeddings
    
    # Lag features
    lags_sequence: List[int] = None  # Lag indices for historical values
    
    # Hardware configuration
    batch_size: int = 1            # Batch size for inference
    dtype: str = "bfloat16"        # Data type
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.lags_sequence is None:
            # Default lags for time series
            self.lags_sequence = [1, 2, 3, 4, 5, 6, 7, 14, 28]
        
        assert self.d_model % self.encoder_attention_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by encoder_attention_heads ({self.encoder_attention_heads})"
        assert self.d_model % self.decoder_attention_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by decoder_attention_heads ({self.decoder_attention_heads})"


def create_model_config(
    context_length: int = 512,
    prediction_length: int = 96,
    input_size: int = 1,
    d_model: int = 128,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    encoder_attention_heads: int = 4,
    decoder_attention_heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 1,
) -> TimeSeriesTransformerConfig:
    """
    Create a Time Series Transformer configuration.
    
    Args:
        context_length: Input context length (default: 512)
        prediction_length: Output prediction length (default: 96)
        input_size: Number of input channels (default: 1)
        d_model: Model dimension (default: 128)
        encoder_layers: Number of encoder layers (default: 2)
        decoder_layers: Number of decoder layers (default: 2)
        encoder_attention_heads: Number of encoder attention heads (default: 4)
        decoder_attention_heads: Number of decoder attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
        batch_size: Batch size (default: 1)
        
    Returns:
        TimeSeriesTransformerConfig instance
    """
    return TimeSeriesTransformerConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        input_size=input_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=decoder_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
    )


def get_default_config() -> TimeSeriesTransformerConfig:
    """
    Get the default Time Series Transformer configuration.
    
    Returns:
        Default TimeSeriesTransformerConfig
    """
    return TimeSeriesTransformerConfig(
        context_length=512,
        prediction_length=96,
        input_size=1,
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        dropout=0.1,
        batch_size=1,
    )


def get_tourism_config() -> TimeSeriesTransformerConfig:
    """
    Get configuration optimized for Tourism dataset.
    
    Returns:
        Tourism-optimized TimeSeriesTransformerConfig
    """
    return TimeSeriesTransformerConfig(
        context_length=512,
        prediction_length=24,
        input_size=1,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        dropout=0.1,
        batch_size=1,
    )


def get_ett_config() -> TimeSeriesTransformerConfig:
    """
    Get configuration optimized for ETT datasets.
    
    Returns:
        ETT-optimized TimeSeriesTransformerConfig
    """
    return TimeSeriesTransformerConfig(
        context_length=512,
        prediction_length=96,
        input_size=7,  # ETT has 7 features
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        dropout=0.1,
        batch_size=1,
    )
