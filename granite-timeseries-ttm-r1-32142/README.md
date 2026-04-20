# Granite Timeseries TTM-R1 (Tiny Time Mixer) Bring-Up Using TTNN APIs

## Issue Information
- **Issue**: [tenstorrent/tt-metal#32142](https://github.com/tenstorrent/tt-metal/issues/32142)
- **Bounty**: $1500
- **Model**: Granite Timeseries TTM-R1 (Tiny Time Mixer) by IBM Research
- **Target Hardware**: Wormhole or Blackhole
- **Pool**: bug

## Overview

This implementation brings up **Granite Timeseries TTM-R1** (Tiny Time Mixer) using TTNN APIs on Tenstorrent hardware.

Granite Timeseries TTM-R1 is a revolutionary compact pre-trained foundation model developed by IBM Research for multivariate time-series forecasting:

- **Ultra-Lightweight**: < 1 million parameters (500x smaller than TimesFM)
- **Pre-trained on 250 million time-series samples**
- **Architecture**: Lightweight MLP-Mixer variant with adaptive patching
- **Optimized for**: Context 512 → Forecast 96
- **Zero-shot and few-shot forecasting capabilities**

## Architecture

The Tiny Time Mixer (TTM) architecture consists of:

1. **Adaptive Patching Layer**: Learns optimal patch size for input
2. **Patch Embedding**: Lightweight projection with positional/channel encoding
3. **Lightweight Time-Mixing Layers**: MLP-Mixer style operations on time dimension
4. **Lightweight Channel-Mixing Layers**: Cross-variate dependency modeling
5. **Residual Connections**: Throughout the network
6. **Normalization Layers**: Efficient LayerNorm
7. **Forecasting Head**: Point prediction output

## Directory Structure

```
granite-timeseries-ttm-r1-32142/
├── README.md                          # This file
├── reference/
│   └── torch_reference.py             # PyTorch reference implementation
├── tt/
│   ├── config.py                      # Model configuration
│   └── model.py                       # TTNN model implementation
├── tests/
│   └── test_model.py                   # Unit tests
├── pydemo/
│   └── demo.py                        # Demo script
└── scripts/
    └── prepare_assets.py              # Asset preparation
```

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | < 1,000,000 |
| Context Length | 512 |
| Forecast Length | 96 |
| Patch Size | 16 |
| Model Dimension (d_model) | 64 |
| Number of Layers | 3 |
| Default Dropout | 0.1 |

## Implementation Stages

### Stage 1 — Bring-Up ✓
- [x] Implement Granite TTM-R1 model using TTNN APIs (Python)
- [x] Implements the Tiny Time Mixer architecture
- [x] Model structure ready for Tenstorrent hardware
- [x] Supports zero-shot and few-shot forecasting architecture
- [ ] Load pre-trained weights from HuggingFace
- [ ] Hardware testing and validation

### Stage 2 — Basic Optimizations
- [ ] Optimal sharded/interleaved memory configs
- [ ] Efficient sharding strategy for MLP-Mixer blocks
- [ ] Fuse simple operations
- [ ] Use recommended TTNN/tt-metal MLP flows

### Stage 3 — Deeper Optimization
- [ ] Maximize core counts
- [ ] Deeper TT-specific optimizations
- [ ] Minimize prediction latency
- [ ] Batch processing for throughput

## Setup

```bash
# Clone the repository (if not already done)
cd /Users/kasion/claude-builders-bounty/granite-timeseries-ttm-r1-32142

# Install dependencies
pip install torch ttnn huggingface-hub

# Prepare assets (if needed)
python scripts/prepare_assets.py
```

## Usage

### PyTorch Reference Model

```python
from reference.torch_reference import create_ttm_r1

# Create model
model = create_ttm_r1(num_channels=7)  # ETT dataset example

# Forward pass
x = torch.randn(1, 7, 512)  # (batch, channels, time)
output = model(x)  # (1, 7, 96) - forecast

print(f"Model parameters: {model.count_parameters():,}")
```

### TTNN Model

```python
import ttnn
from tt.config import GraniteTTMConfig, create_model_config
from tt.model import create_ttm_r1

# Initialize device
device = ttnn.open_device(device_id=0)

# Create configuration
config = create_model_config(
    context_length=512,
    forecast_length=96,
    num_channels=7,
    d_model=64,
    patch_size=16,
    num_layers=3,
)

# Create TTNN model (requires preprocessed weights)
model = create_ttm_r1(
    device=device,
    config=config,
    parameters={},  # Preprocessed weights
    dtype=ttnn.bfloat16,
)

# Run inference
# ... (full implementation with preprocessing)
```

## Testing

```bash
# Run unit tests
pytest tests/test_model.py -v

# Run demo
python pydemo/demo.py
```

## References

- **HuggingFace Model**: [ibm-granite/granite-timeseries-ttm-r1](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
- **IBM TSFM Repository**: https://github.com/IBM/tsfm
- **Research Paper**: "Tiny Time Mixers (TTM): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting" (arXiv:2401.03955)
- **TTNN Model Bring-up Tech Report**: See tt-metal tech_reports

## Related Implementations

- [Informer Time-Series Model](../informer/) - Another TTNN time-series implementation in tt-metal
- [PatchTSMixer](../patch_tsmixer/) - Larger IBM time-series model (similar architecture)

## Notes

- This is a bring-up implementation for the Granite Timeseries TTM-R1 model
- The model is extremely lightweight (< 1M parameters), enabling high throughput
- Target: 500+ sequences/second inference throughput
- Optimized for edge deployment and multi-tenant scenarios
