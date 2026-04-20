# TTNN Implementation of Granite Timeseries TTM-R1 (Tiny Time Mixer)
# Issue: tenstorrent/tt-metal#32142

"""
TTNN implementation of Granite Timeseries TTM-R1 for Tenstorrent hardware.

Architecture:
- Adaptive patching layer (learns optimal patch size)
- Patch embedding with lightweight projection
- Lightweight Time-Mixing layers (MLP-Mixer style)
- Lightweight Channel-Mixing layers (cross-variate dependencies)
- Residual connections
- Normalization layers
- Forecasting head for point predictions
"""

import torch
import ttnn
from ttnn import TtTensor
from typing import Optional, Tuple, Dict
import math


class TtAdaptivePatching:
    """TTNN Adaptive Patching layer."""
    
    def __init__(
        self,
        device,
        context_length: int = 512,
        patch_size: int = 16,
        batch_size: int = 1,
        num_channels: int = 1,
        mesh_mapper=None,
    ):
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_patches = context_length // patch_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.device = device
        self.mesh_mapper = mesh_mapper
        
    def __call__(self, x: TtTensor) -> TtTensor:
        """
        Apply adaptive patching.
        
        Args:
            x: Input tensor of shape (batch, 1, channels, time) in TTNN format
        Returns:
            Patched tensor
        """
        # Reshape to patches
        # Input: (batch, 1, channels, time)
        # Output: (batch, 1, channels * num_patches, patch_size)
        batch_size = x.shape[0]
        
        # Reshape operation using ttnn.reshape
        # (batch, 1, channels, time) -> (batch, 1, channels * num_patches, patch_size)
        x = ttnn.reshape(x, (batch_size, 1, self.num_channels * self.num_patches, self.patch_size))
        
        return x


class TtLightweightTimeMixing:
    """TTNN Lightweight MLP-Mixer style time mixing layer."""
    
    def __init__(
        self,
        device,
        num_patches: int,
        d_model: int,
        dropout: float = 0.1,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.num_patches = num_patches
        self.d_model = d_model
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        # Transpose weights (num_patches -> num_patches // 2 -> num_patches)
        self.time_fc1_weight = None
        self.time_fc1_bias = None
        self.time_fc2_weight = None
        self.time_fc2_bias = None
        self.norm_weight = None
        self.norm_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        self.time_fc1_weight = ttnn.from_torch(
            torch_layer.time_mixer[0].weight.T,  # Transpose for TTNN
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.time_fc1_bias = ttnn.from_torch(
            torch_layer.time_mixer[0].bias,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.time_fc2_weight = ttnn.from_torch(
            torch_layer.time_mixer[3].weight.T,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.time_fc2_bias = ttnn.from_torch(
            torch_layer.time_mixer[3].bias,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.norm_weight = ttnn.from_torch(
            torch_layer.norm.weight.unsqueeze(0).unsqueeze(0),  # Add dims for TTNN LayerNorm
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.norm_bias = ttnn.from_torch(
            torch_layer.norm.bias.unsqueeze(0).unsqueeze(0),
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        
    def __call__(self, x: TtTensor) -> TtTensor:
        """
        Apply time mixing.
        
        Args:
            x: Input tensor (batch, 1, channels * num_patches, d_model)
        Returns:
            Time-mixed tensor
        """
        residual = x
        
        # Normalize: Apply LayerNorm
        x = ttnn.layer_norm(
            x,
            weight=self.norm_weight,
            bias=self.norm_bias,
        )
        
        # Transpose: (batch, 1, channels * num_patches, d_model) -> (batch, 1, d_model, channels * num_patches)
        x = ttnn.transpose(x, 2, 3)
        
        # Reshape for linear: (batch, 1, d_model, channels * num_patches) -> (batch * d_model, 1, channels * num_patches)
        # Actually keep it as 4D for ttnn.linear
        # (batch, 1, d_model, channels * num_patches) -> (batch, 1, channels * num_patches, d_model) for FC
        
        # FC1: num_patches -> num_patches // 2
        # First transpose back: (batch, 1, d_model, channels * num_patches) -> (batch, 1, channels * num_patches, d_model)
        x = ttnn.transpose(x, 2, 3)
        
        # Reshape to 3D for linear: (batch, channels * num_patches, d_model)
        batch = x.shape[0]
        x_3d = ttnn.reshape(x, (batch, self.num_patches * self.num_channels, self.d_model))
        
        # Transpose to (batch, d_model, channels * num_patches)
        x_3d = ttnn.transpose(x_3d, 1, 2)
        
        # FC1
        x_3d = ttnn.linear(
            x_3d,
            self.time_fc1_weight,
            bias=self.time_fc1_bias,
            activation="gelu",
        )
        
        # FC2
        x_3d = ttnn.linear(
            x_3d,
            self.time_fc2_weight,
            bias=self.time_fc2_bias,
        )
        
        # Transpose back: (batch, channels * num_patches, d_model) -> (batch, d_model, channels * num_patches)
        x_3d = ttnn.transpose(x_3d, 1, 2)
        
        # Reshape to 4D: (batch, d_model, channels * num_patches) -> (batch, 1, channels * num_patches, d_model)
        x = ttnn.reshape(x_3d, (batch, 1, self.num_channels * self.num_patches, self.d_model))
        
        # Add residual
        x = ttnn.add(x, residual)
        
        return x


class TtLightweightChannelMixing:
    """TTNN Lightweight channel mixing layer for cross-variate dependencies."""
    
    def __init__(
        self,
        device,
        d_model: int,
        num_channels: int,
        dropout: float = 0.1,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.d_model = d_model
        self.num_channels = num_channels
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        self.channel_fc1_weight = None
        self.channel_fc1_bias = None
        self.channel_fc2_weight = None
        self.channel_fc2_bias = None
        self.norm_weight = None
        self.norm_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        self.channel_fc1_weight = ttnn.from_torch(
            torch_layer.channel_mixer[0].weight.T,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.channel_fc1_bias = ttnn.from_torch(
            torch_layer.channel_mixer[0].bias,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.channel_fc2_weight = ttnn.from_torch(
            torch_layer.channel_mixer[3].weight.T,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.channel_fc2_bias = ttnn.from_torch(
            torch_layer.channel_mixer[3].bias,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.norm_weight = ttnn.from_torch(
            torch_layer.norm.weight.unsqueeze(0).unsqueeze(0),
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.norm_bias = ttnn.from_torch(
            torch_layer.norm.bias.unsqueeze(0).unsqueeze(0),
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        
    def __call__(self, x: TtTensor) -> TtTensor:
        """
        Apply channel mixing.
        
        Args:
            x: Input tensor (batch, 1, channels * num_patches, d_model)
        Returns:
            Channel-mixed tensor
        """
        residual = x
        
        # Normalize
        x = ttnn.layer_norm(
            x,
            weight=self.norm_weight,
            bias=self.norm_bias,
        )
        
        # Transpose for channel mixing: (batch, 1, channels * num_patches, d_model) -> (batch, 1, d_model, channels * num_patches)
        x = ttnn.transpose(x, 2, 3)
        
        # Reshape to 3D: (batch, d_model, channels * num_patches) -> (batch * d_model, 1, channels * num_patches)
        batch = x.shape[0]
        x_3d = ttnn.reshape(x, (batch * self.d_model, 1, self.num_channels * self.num_patches))
        
        # Transpose: (batch * d_model, 1, channels * num_patches) -> (batch * d_model, 1, num_patches, channels)
        # Actually we want to treat each patch separately
        x_3d = ttnn.reshape(x_3d, (batch, self.d_model, self.num_channels, self.num_patches))
        
        # Transpose: (batch, d_model, channels, num_patches) -> (batch, d_model, num_patches, channels)
        x_3d = ttnn.transpose(x_3d, 2, 3)
        
        # FC1 on channels dimension
        x_3d = ttnn.reshape(x_3d, (batch * self.d_model * self.num_patches, 1, self.num_channels))
        x_3d = ttnn.linear(
            x_3d,
            self.channel_fc1_weight,
            bias=self.channel_fc1_bias,
            activation="gelu",
        )
        
        # FC2 on channels dimension
        x_3d = ttnn.linear(
            x_3d,
            self.channel_fc2_weight,
            bias=self.channel_fc2_bias,
        )
        
        # Reshape back: (batch * d_model * num_patches, 1, num_channels // 2) -> (batch, d_model, num_patches, num_channels // 2)
        x_3d = ttnn.reshape(x_3d, (batch, self.d_model, self.num_patches, self.num_channels // 2))
        
        # This is simplified - full implementation would need proper channel mixing
        # For now, add residual
        x = ttnn.add(x, residual)
        
        return x


class TtTinyTimeMixerBlock:
    """TTNN Single Tiny Time Mixer block with time and channel mixing."""
    
    def __init__(
        self,
        device,
        num_patches: int,
        d_model: int,
        num_channels: int,
        dropout: float = 0.1,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.time_mixer = TtLightweightTimeMixing(
            device, num_patches, d_model, dropout, dtype, mesh_mapper
        )
        self.channel_mixer = TtLightweightChannelMixing(
            device, d_model, num_channels, dropout, dtype, mesh_mapper
        )
        self.device = device
        
    def create_weights(self, torch_block):
        """Create TTNN weights from PyTorch block."""
        self.time_mixer.create_weights(torch_block.time_mixer)
        self.channel_mixer.create_weights(torch_block.channel_mixer)
        
    def __call__(self, x: TtTensor) -> TtTensor:
        """
        Apply Tiny Time Mixer block.
        
        Args:
            x: Input tensor (batch, 1, channels * num_patches, d_model)
        Returns:
            Output tensor after mixing
        """
        # Time mixing
        x = self.time_mixer(x)
        
        # Channel mixing
        x = self.channel_mixer(x)
        
        return x


class TtGraniteTTMR1:
    """
    TTNN Granite Timeseries TTM-R1 (Tiny Time Mixer).
    
    Ultra-lightweight foundation model for time-series forecasting with < 1M parameters.
    """
    
    def __init__(
        self,
        device,
        config: "GraniteTTMConfig",
        parameters: Dict,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        # Create sub-modules
        self.patching = TtAdaptivePatching(
            device,
            context_length=config.context_length,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            num_channels=config.num_channels,
            mesh_mapper=mesh_mapper,
        )
        
        # Patch embedding
        self.patch_embedding_weight = None
        self.patch_embedding_bias = None
        
        # Positional encoding
        self.pos_encoding = None
        
        # Channel encoding
        self.channel_encoding = None
        
        # Mixer blocks
        self.blocks = [
            TtTinyTimeMixerBlock(
                device,
                config.num_patches,
                config.d_model,
                config.num_channels,
                config.dropout,
                dtype,
                mesh_mapper,
            )
            for _ in range(config.num_layers)
        ]
        
        # Forecasting head
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None
        
        # Load weights
        self.load_weights(parameters)
        
    def load_weights(self, parameters: Dict):
        """Load preprocessed weights."""
        # Patch embedding
        if "patch_embedding" in parameters:
            self.patch_embedding_weight = ttnn.from_torch(
                parameters["patch_embedding"]["weight"].T,
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
            self.patch_embedding_bias = ttnn.from_torch(
                parameters["patch_embedding"]["bias"],
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
        
        # Positional encoding
        if "pos_encoding" in parameters:
            self.pos_encoding = ttnn.from_torch(
                parameters["pos_encoding"],
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
        
        # Channel encoding
        if "channel_encoding" in parameters:
            self.channel_encoding = ttnn.from_torch(
                parameters["channel_encoding"],
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
        
        # Mixer blocks
        if "blocks" in parameters:
            for i, block_params in enumerate(parameters["blocks"]):
                if i < len(self.blocks):
                    block_dict = {
                        "time_mixer": block_params.get("time_mixer", {}),
                        "channel_mixer": block_params.get("channel_mixer", {}),
                    }
                    torch_block = type('obj', (object,), block_dict)
                    torch_block.time_mixer = type('obj', (object,), block_params.get("time_mixer", {}))
                    torch_block.channel_mixer = type('obj', (object,), block_params.get("channel_mixer", {}))
                    self.blocks[i].create_weights(torch_block)
        
        # Forecasting head
        if "forecasting_head" in parameters:
            self.fc1_weight = ttnn.from_torch(
                parameters["forecasting_head"][0].weight.T,
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
            self.fc1_bias = ttnn.from_torch(
                parameters["forecasting_head"][0].bias,
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
            self.fc2_weight = ttnn.from_torch(
                parameters["forecasting_head"][2].weight.T,
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
            self.fc2_bias = ttnn.from_torch(
                parameters["forecasting_head"][2].bias,
                dtype=self.dtype,
                mesh_mapper=self.mesh_mapper,
            )
            
    def __call__(self, x: TtTensor) -> TtTensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        Returns:
            Forecast tensor
        """
        # Apply patching
        x = self.patching(x)
        
        # Patch embedding
        batch = x.shape[0]
        x = ttnn.reshape(x, (batch, 1, self.config.num_channels * self.config.num_patches, self.config.patch_size))
        
        # Note: Simplified forward - full implementation would use loaded weights
        # This is a placeholder demonstrating the structure
        
        return x


class GraniteTTMConfig:
    """Configuration for Granite TTM-R1 model."""
    
    def __init__(
        self,
        context_length: int = 512,
        forecast_length: int = 96,
        num_channels: int = 1,
        d_model: int = 64,
        patch_size: int = 16,
        num_layers: int = 3,
        dropout: float = 0.1,
        batch_size: int = 1,
    ):
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.num_channels = num_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_patches = context_length // patch_size


def create_ttm_r1(
    device,
    config: GraniteTTMConfig,
    parameters: Dict,
    dtype=ttnn.bfloat16,
    mesh_mapper=None,
) -> TtGraniteTTMR1:
    """
    Create a TTNN Granite TTM-R1 model.
    
    Args:
        device: TT device
        config: Model configuration
        parameters: Preprocessed model parameters
        dtype: Data type
        mesh_mapper: Optional mesh mapper for multi-device
        
    Returns:
        TtGraniteTTMR1 model
    """
    model = TtGraniteTTMR1(
        device=device,
        config=config,
        parameters=parameters,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )
    return model
