# Time Series Transformer TTNN Implementation

**Issue**: tenstorrent/tt-metal#32140  
**Bounty**: $1500  
**Status**: Implementation Complete

## Overview

This is a TTNN (Tenstorrent Neural Network) implementation of the Time Series Transformer model for probabilistic time-series forecasting on Tenstorrent hardware.

### Architecture

The Time Series Transformer is a vanilla encoder-decoder Transformer architecture specifically designed for probabilistic time-series forecasting. Based on the HuggingFace Transformers implementation.

**Key Components:**
- **Value Embedding Layer**: Projects historical values with optional lag features
- **Temporal Feature Embeddings**: Encodes past and future time features
- **Static Feature Embeddings**: Handles categorical and real static features
- **Transformer Encoder**: Self-attention over historical context
- **Transformer Decoder**: 
  - Masked self-attention (causal masking)
  - Cross-attention to encoder outputs
- **Distribution Head**: Probabilistic outputs (Student-t, Normal, or Negative Binomial)

### Features

- **Probabilistic Forecasting**: Outputs distribution parameters instead of point estimates
- **Multiple Distribution Types**: Student-t (default), Normal, Negative Binomial
- **Rich Feature Support**: 
  - Past values (historical time series)
  - Temporal features (hour, day, weekday, month, etc.)
  - Static categorical features (store ID, region, etc.)
  - Static real features (store size, product price, etc.)
  - Lag features for capturing historical patterns
- **Teacher-Forcing Training**: Efficient training paradigm
- **Autoregressive Generation**: Flexible inference with multiple samples

## Implementation Status

### Stage 1 — Bring-Up ✅
- [x] TTNN model implementation using Python APIs
- [x] Full encoder-decoder architecture
- [x] Support for multiple distribution outputs
- [x] Support for comprehensive feature inputs
- [x] Documentation and reference implementation

### Stage 2 — Basic Optimizations (Planned)
- [ ] Optimal sharded/interleaved memory configs
- [ ] Efficient sharding strategy for attention
- [ ] Operation fusion opportunities
- [ ] TT library integration

### Stage 3 — Deeper Optimization (Planned)
- [ ] Flash Attention or equivalent
- [ ] Optimized KV-cache management
- [ ] Pipeline encoder and decoder
- [ ] Advanced performance tuning

## File Structure

```
time-series-transformer-32140/
├── README.md                    # This file
├── tt/
│   ├── __init__.py             # Package init
│   ├── config.py               # Configuration module
│   └── model.py                # TTNN model implementation
├── reference/
│   └── torch_reference.py       # PyTorch reference implementation
└── tests/
    └── test_model.py            # Basic tests
```

## Usage

### PyTorch Reference Model

```python
import torch
from reference.torch_reference import create_model

# Create model
model = create_model(
    context_length=512,
    prediction_length=96,
    input_size=1,
    d_model=128,
)

# Create inputs
past_values = torch.randn(1, 512, 1)
past_time_features = torch.randn(1, 512, 5)
future_time_features = torch.randn(1, 96, 5)

# Forward pass
dist_params = model(past_values, past_time_features, future_time_features)
```

### TTNN Model (Hardware Acceleration)

```python
import ttnn
from tt.model import TtTimeSeriesTransformer
from tt.config import get_default_config

# Initialize device
device = ttnn.open_device(device_id=0)

# Create config and model
config = get_default_config()
model = TtTimeSeriesTransformer(device, config, parameters, dtype=ttnn.bfloat16)

# Prepare inputs (convert from PyTorch)
past_values_tt = ttnn.from_torch(past_values, dtype=ttnn.bfloat16)
past_time_features_tt = ttnn.from_torch(past_time_features, dtype=ttnn.bfloat16)
future_time_features_tt = ttnn.from_torch(future_time_features, dtype=ttnn.bfloat16)

# Forward pass
dist_params = model(past_values_tt, past_time_features_tt, future_time_features_tt)
```

## Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_length` | 512 | Input context length |
| `prediction_length` | 96 | Output prediction length |
| `input_size` | 1 | Number of target variables |
| `d_model` | 128 | Model dimension |
| `encoder_layers` | 2 | Number of encoder layers |
| `decoder_layers` | 2 | Number of decoder layers |
| `encoder_attention_heads` | 4 | Number of attention heads |
| `decoder_attention_heads` | 4 | Number of attention heads |
| `encoder_ffn_dim` | 512 | Encoder feed-forward dimension |
| `decoder_ffn_dim` | 512 | Decoder feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `distribution_output` | "student_t" | Distribution type |

## Distribution Outputs

### Student-t Distribution (Default)
- Handles outliers well
- Parameters: df (degrees of freedom), loc, scale

### Normal Distribution
- Standard Gaussian
- Parameters: loc, scale

### Negative Binomial Distribution
- For count data (e.g., sales)
- Parameters: logits

## Performance Targets

### Stage 1 Targets
- Model runs on Tenstorrent hardware without errors
- Inference throughput: ≥100 sequences/second
- Latency: <50ms for single sequence (batch size 1)
- Sample generation: 100 samples in <1 second

### Stage 3 Stretched Goals
- 500+ sequences/second throughput
- <20ms latency for single sequence
- 1000+ samples in <2 seconds
- Context lengths up to 2048

## References

- [HuggingFace Time Series Transformer Documentation](https://huggingface.co/docs/transformers/en/model_doc/time_series_transformer)
- [HuggingFace Blog Post: Probabilistic Time Series Forecasting](https://huggingface.co/blog/time-series-transformers)
- [TTNN Model Bring-up Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- [TT Fused Ops PR #29236](https://github.com/tenstorrent/tt-metal/pull/29236)

## Datasets

Standard benchmarks for evaluation:
- **Tourism Dataset**: Monthly tourism data
- **ETT (Electricity Transformer Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Weather Dataset**: 21 meteorological indicators
- **Traffic Dataset**: Road occupancy rates
- **Electricity (ECL) Dataset**: Hourly electricity consumption

## License

MIT License - See LICENSE file for details
