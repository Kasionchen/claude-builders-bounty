# Time Series Transformer TTNN Implementation
# Issue: tenstorrent/tt-metal#32140
# Bounty: $1500

"""TTNN implementation of Time Series Transformer for Tenstorrent hardware."""

from tt.model import (
    TtTimeSeriesTransformer,
    TtValueEmbedding,
    TtTemporalEmbedding,
    TtStaticFeatureEmbedding,
    TtTransformerEncoderLayer,
    TtTransformerDecoderLayer,
    TtDistributionHead,
)
from tt.config import (
    TimeSeriesTransformerConfig,
    create_model_config,
    get_default_config,
    get_tourism_config,
    get_ett_config,
)

__all__ = [
    "TtTimeSeriesTransformer",
    "TtValueEmbedding",
    "TtTemporalEmbedding",
    "TtStaticFeatureEmbedding",
    "TtTransformerEncoderLayer",
    "TtTransformerDecoderLayer",
    "TtDistributionHead",
    "TimeSeriesTransformerConfig",
    "create_model_config",
    "get_default_config",
    "get_tourism_config",
    "get_ett_config",
]
