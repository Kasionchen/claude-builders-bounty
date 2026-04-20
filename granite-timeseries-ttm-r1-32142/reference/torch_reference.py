# Reference: IBM Granite Timeseries TTM-R1 (Tiny Time Mixer)
# Paper: https://arxiv.org/abs/2401.03955
# Model: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1

"""
Granite Timeseries TTM-R1 is a tiny pre-trained foundation model for time-series forecasting.
Architecture: Lightweight MLP-Mixer variant with:
- Adaptive patching layer
- Lightweight time-mixing layers (MLP-Mixer style)
- Lightweight channel-mixing layers (cross-variate dependencies)
- Residual connections
- Normalization layers
- Forecasting head

Specifications:
- Parameters: < 1 million
- Context length: 512
- Forecast length: 96
- Resolution: Minutely to hourly (10 min, 15 min, 1 hour)
- Type: Point forecasting (not probabilistic)
"""

import torch
import torch.nn as nn
import math


class AdaptivePatching(nn.Module):
    """Learnable patch size for adaptive patching."""
    
    def __init__(self, context_length: int = 512, patch_size: int = 16):
        super().__init__()
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_patches = context_length // patch_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time) or (batch, time, channels)
        Returns:
            Patched tensor of shape (batch, channels, num_patches, patch_size)
        """
        # Reshape to patches
        if x.dim() == 3:
            batch, channels, time = x.shape
            x = x.reshape(batch, channels, self.num_patches, self.patch_size)
        return x


class LightweightTimeMixing(nn.Module):
    """Lightweight MLP-Mixer style time mixing layer."""
    
    def __init__(self, num_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        
        # Lightweight mlp: smaller hidden dimension
        self.time_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches // 2),  # Transpose
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches // 2, num_patches),
        )
        self.norm = nn.LayerNorm([num_patches, d_model])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, patches, d_model) or (batch, patches, d_model)
        Returns:
            Time-mixed tensor
        """
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Transpose for time mixing: (batch, patches, d_model)
        x = x.transpose(-2, -3)
        
        # Apply lightweight mlp
        x = self.time_mixer(x)
        
        # Transpose back
        x = x.transpose(-2, -3)
        
        return x + residual


class LightweightChannelMixing(nn.Module):
    """Lightweight channel mixing layer for cross-variate dependencies."""
    
    def __init__(self, d_model: int, num_channels: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        
        # Lightweight mlp for channel mixing
        self.channel_mixer = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels // 2, num_channels),
        )
        self.norm = nn.LayerNorm([num_channels, d_model])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, patches, d_model) or (batch, patches, d_model)
        Returns:
            Channel-mixed tensor
        """
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Transpose for channel mixing: (batch, d_model, patches, channels) -> (batch, patches, channels, d_model)
        x = x.transpose(-1, -2)
        
        # Apply lightweight mlp
        x = self.channel_mixer(x)
        
        # Transpose back
        x = x.transpose(-1, -2)
        
        return x + residual


class TinyTimeMixerBlock(nn.Module):
    """Single Tiny Time Mixer block with time and channel mixing."""
    
    def __init__(self, num_patches: int, d_model: int, num_channels: int, dropout: float = 0.1):
        super().__init__()
        self.time_mixer = LightweightTimeMixing(num_patches, d_model, dropout)
        self.channel_mixer = LightweightChannelMixing(d_model, num_channels, dropout)
        self.norm = nn.LayerNorm([num_channels, num_patches, d_model])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, patches, d_model)
        Returns:
            Output tensor after mixing
        """
        # Time mixing (works on patches dimension)
        x = x + self.time_mixer(x)
        
        # Channel mixing (works on channels dimension)
        x = x + self.channel_mixer(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class GraniteTTMR1(nn.Module):
    """
    Granite Timeseries TTM-R1 (Tiny Time Mixer).
    
    Ultra-lightweight foundation model for time-series forecasting with < 1M parameters.
    Optimized for:
    - Context length: 512
    - Forecast length: 96
    - Zero-shot and few-shot forecasting
    """
    
    def __init__(
        self,
        context_length: int = 512,
        forecast_length: int = 96,
        num_channels: int = 1,
        d_model: int = 64,
        patch_size: int = 16,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.num_channels = num_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = context_length // patch_size
        
        # Adaptive patching
        self.patching = AdaptivePatching(context_length, patch_size)
        
        # Patch embedding with lightweight projection
        self.patch_embedding = nn.Linear(patch_size, d_model)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # Channel encoding for multivariate time series
        self.channel_encoding = nn.Parameter(torch.randn(1, num_channels, 1, d_model) * 0.02)
        
        # Stack of Tiny Time Mixer blocks
        self.blocks = nn.ModuleList([
            TinyTimeMixerBlock(self.num_patches, d_model, num_channels, dropout)
            for _ in range(num_layers)
        ])
        
        # Forecasting head
        self.forecasting_head = nn.Sequential(
            nn.Linear(d_model * self.num_patches, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, forecast_length * num_channels),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following transformer conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_loss: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time) or (batch, time, channels)
            return_loss: If True, also compute MSE loss against future values
            
        Returns:
            Forecast tensor of shape (batch, channels, forecast_length)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, time) -> (batch, 1, time)
        elif x.dim() == 2 and x.shape[-1] == self.num_channels:
            x = x.unsqueeze(1)  # (batch, channels) -> (batch, channels, 1)
        
        batch_size, channels, time = x.shape
        
        # Adaptive patching: (batch, channels, time) -> (batch, channels, num_patches, patch_size)
        x = self.patching(x)
        
        # Patch embedding: (batch, channels, num_patches, patch_size) -> (batch, channels, num_patches, d_model)
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(1)  # Broadcast to (batch, channels, num_patches, d_model)
        
        # Add channel encoding
        x = x + self.channel_encoding
        
        # Apply Tiny Time Mixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Flatten for forecasting head: (batch, channels, num_patches, d_model) -> (batch, channels, num_patches * d_model)
        x = x.reshape(batch_size, channels, self.num_patches * self.d_model)
        
        # Forecast head: (batch, channels, num_patches * d_model) -> (batch, channels, forecast_length)
        x = self.forecasting_head(x)
        x = x.reshape(batch_size, channels, self.forecast_length)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ttm_r1(num_channels: int = 1, pretrained: bool = False) -> GraniteTTMR1:
    """
    Create a Granite TTM-R1 model.
    
    Args:
        num_channels: Number of input channels (variables)
        pretrained: If True, loads pretrained weights (not implemented yet)
        
    Returns:
        GraniteTTMR1 model
    """
    model = GraniteTTMR1(
        context_length=512,
        forecast_length=96,
        num_channels=num_channels,
        d_model=64,
        patch_size=16,
        num_layers=3,
        dropout=0.1,
    )
    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    context_length = 512
    num_channels = 7  # Example: ETT dataset
    forecast_length = 96
    
    model = create_ttm_r1(num_channels=num_channels)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Expected: < 1,000,000")
    
    # Create sample input
    x = torch.randn(batch_size, num_channels, context_length)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_channels}, {forecast_length})")
    assert output.shape == (batch_size, num_channels, forecast_length), "Output shape mismatch!"
    print("✓ Model test passed!")
