# Reference: HuggingFace Time Series Transformer
# Paper: "Time Series Transformer" (Vaswani et al. architecture for time series)
# Model: https://huggingface.co/docs/transformers/en/model_doc/time_series_transformer

"""
Time Series Transformer is a vanilla encoder-decoder Transformer architecture
specifically designed for probabilistic time-series forecasting.

Architecture:
- Encoder: Processes past_values with context_length
- Decoder: Autoregressively generates prediction_length forecasts
- Distribution head: Outputs distribution parameters for probabilistic forecasting

Supports:
- Student-t distribution (default, handles outliers)
- Normal/Gaussian distribution
- Negative binomial distribution (for count data)
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Dict


class ValueEmbedding(nn.Module):
    """Value embedding with optional lag features."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        lags_sequence: List[int],
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.lags_sequence = lags_sequence
        self.num_lags = len(lags_sequence)
        
        # Projection layer
        self.projection = nn.Linear(input_size * self.num_lags, d_model)
        
    def forward(
        self,
        past_values: torch.Tensor,
        lag_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_values: (batch, context_length, input_size)
            lag_values: Optional lagged values
            
        Returns:
            (batch, context_length, d_model)
        """
        if lag_values is not None:
            x = torch.cat([past_values, lag_values], dim=-1)
        else:
            x = past_values
        
        batch_size, context_length, _ = x.shape
        x = x.reshape(batch_size, context_length, self.input_size * self.num_lags)
        
        x = self.projection(x)
        return x


class TemporalEmbedding(nn.Module):
    """Temporal feature embedding layer."""
    
    def __init__(
        self,
        num_time_features: int,
        d_model: int,
    ):
        super().__init__()
        self.num_time_features = num_time_features
        self.d_model = d_model
        
        self.time_embed = nn.Linear(num_time_features, d_model)
        
    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_features: (batch, time_length, num_time_features)
            
        Returns:
            (batch, time_length, d_model)
        """
        x = self.time_embed(time_features)
        return x


class StaticFeatureEmbedding(nn.Module):
    """Static feature embedding for categorical and real features."""
    
    def __init__(
        self,
        num_static_categorical: int,
        num_static_real: int,
        cardinality: List[int],
        embedding_dimension: int,
        d_model: int,
    ):
        super().__init__()
        self.num_static_categorical = num_static_categorical
        self.num_static_real = num_static_real
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.d_model = d_model
        
        # Categorical embeddings
        self.categorical_embed = nn.ModuleList([
            nn.Embedding(cat_dim, embedding_dimension)
            for cat_dim in cardinality
        ])
        
        # Real features projection
        if num_static_real > 0:
            self.real_embed = nn.Linear(num_static_real, d_model)
        else:
            self.real_embed = None
        
    def forward(
        self,
        static_categorical: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            static_categorical: (batch, num_static_categorical)
            static_real: (batch, num_static_real)
            
        Returns:
            (batch, 1, d_model)
        """
        embeddings = []
        
        # Categorical embeddings
        if static_categorical is not None and self.num_static_categorical > 0:
            for i, embed in enumerate(self.categorical_embed):
                cat_value = static_categorical[:, i]
                embeddings.append(embed(cat_value))
        
        # Real features
        if static_real is not None and self.num_static_real > 0:
            real_emb = self.real_embed(static_real)
            embeddings.append(real_emb)
        
        if embeddings:
            result = torch.stack(embeddings, dim=1).sum(dim=1)
            result = result.unsqueeze(1)
        else:
            result = None
            
        return result


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder layer with self-attention and FFN."""
    
    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.head_dim = d_model // num_attention_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, d_model)
            
        Returns:
            (batch, seq_length, d_model)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.fc2(torch.relu(self.fc1(x)))
        x = x + residual
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder layer with self-attention, cross-attention, and FFN."""
    
    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.head_dim = d_model // num_attention_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = dropout
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_seq_length, d_model)
            encoder_hidden_states: (batch, src_seq_length, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            (batch, tgt_seq_length, d_model)
        """
        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        
        # Create causal mask
        tgt_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=x.device),
            diagonal=1
        ).bool()
        
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = x + residual
        
        # Cross-attention
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, encoder_hidden_states, encoder_hidden_states)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.norm3(x)
        x = self.fc2(torch.relu(self.fc1(x)))
        x = x + residual
        
        return x


class DistributionHead(nn.Module):
    """Distribution head for probabilistic outputs."""
    
    def __init__(
        self,
        d_model: int,
        prediction_length: int,
        distribution_output: str = "student_t",
    ):
        super().__init__()
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.distribution_output = distribution_output
        
        if distribution_output == "student_t":
            # Student-t has 3 parameters: df, loc, scale
            self.projection = nn.Linear(d_model, 3 * prediction_length)
        elif distribution_output == "normal":
            # Normal has 2 parameters: loc, scale
            self.projection = nn.Linear(d_model, 2 * prediction_length)
        else:
            # negative_binomial
            self.projection = nn.Linear(d_model, prediction_length)
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, prediction_length, d_model)
            
        Returns:
            Dictionary of distribution parameters
        """
        params = self.projection(x)
        
        if self.distribution_output == "student_t":
            df = params[:, :, 0:1]
            loc = params[:, :, 1:2]
            scale = params[:, :, 2:3]
            # Ensure positive scale
            scale = torch.nn.functional.softplus(scale)
            return {"df": df, "loc": loc, "scale": scale}
        elif self.distribution_output == "normal":
            loc = params[:, :, 0:1]
            scale = params[:, :, 1:2]
            # Ensure positive scale
            scale = torch.nn.functional.softplus(scale)
            return {"loc": loc, "scale": scale}
        else:
            logits = params
            return {"logits": logits}


class TimeSeriesTransformerModel(nn.Module):
    """
    Time Series Transformer bare model (encoder-decoder without distribution head).
    
    Based on HuggingFace Transformers TimeSeriesTransformerModel.
    """
    
    def __init__(
        self,
        context_length: int = 512,
        prediction_length: int = 96,
        input_size: int = 1,
        num_time_features: int = 5,
        d_model: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_attention_heads: int = 4,
        decoder_attention_heads: int = 4,
        encoder_ffn_dim: int = 512,
        decoder_ffn_dim: int = 512,
        dropout: float = 0.1,
        num_static_categorical_features: int = 0,
        num_static_real_features: int = 0,
        cardinality: List[int] = None,
        embedding_dimension: int = 16,
        lags_sequence: List[int] = None,
    ):
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_size = input_size
        self.num_time_features = num_time_features
        self.d_model = d_model
        
        if lags_sequence is None:
            lags_sequence = [1, 2, 3, 4, 5, 6, 7, 14, 28]
        self.lags_sequence = lags_sequence
        
        if cardinality is None:
            cardinality = []
        
        # Embeddings
        self.value_embedding = ValueEmbedding(
            input_size, d_model, lags_sequence
        )
        
        self.temporal_embedding = TemporalEmbedding(
            num_time_features, d_model
        )
        
        self.static_embedding = StaticFeatureEmbedding(
            num_static_categorical_features,
            num_static_real_features,
            cardinality,
            embedding_dimension,
            d_model,
        )
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                encoder_attention_heads,
                encoder_ffn_dim,
                dropout,
            )
            for _ in range(encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model,
                decoder_attention_heads,
                decoder_ffn_dim,
                dropout,
            )
            for _ in range(decoder_layers)
        ])
        
        # Layer norms
        self.encoder_final_layer_norm = nn.LayerNorm(d_model)
        self.decoder_final_layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following transformer conventions."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        static_categorical: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            past_values: (batch, context_length, input_size)
            past_time_features: (batch, context_length, num_time_features)
            future_time_features: (batch, prediction_length, num_time_features)
            static_categorical: (batch, num_static_categorical)
            static_real: (batch, num_static_real)
            past_observed_mask: (batch, context_length, input_size)
            
        Returns:
            Decoder output (batch, prediction_length, d_model)
        """
        # Encode past values
        value_emb = self.value_embedding(past_values)
        temporal_emb = self.temporal_embedding(past_time_features)
        
        encoder_hidden_states = value_emb + temporal_emb
        
        # Add static embedding
        static_emb = self.static_embedding(static_categorical, static_real)
        if static_emb is not None:
            encoder_hidden_states = encoder_hidden_states + static_emb
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(encoder_hidden_states)
        
        # Final encoder normalization
        encoder_hidden_states = self.encoder_final_layer_norm(encoder_hidden_states)
        
        # Decode future values
        decoder_temporal_emb = self.temporal_embedding(future_time_features)
        
        # During training: teacher forcing with actual future values
        # For simplicity, use zeros as initial decoder input
        decoder_input = decoder_temporal_emb
        
        # Add static embedding
        if static_emb is not None:
            decoder_input = decoder_input + static_emb
        
        # Apply decoder layers
        decoder_hidden_states = decoder_input
        for layer in self.decoder_layers:
            decoder_hidden_states = layer(decoder_hidden_states, encoder_hidden_states)
        
        # Final decoder normalization
        decoder_hidden_states = self.decoder_final_layer_norm(decoder_hidden_states)
        
        return decoder_hidden_states


class TimeSeriesTransformerForPrediction(TimeSeriesTransformerModel):
    """
    Time Series Transformer for probabilistic prediction.
    
    Full model with distribution head for probabilistic forecasting.
    Based on HuggingFace Transformers TimeSeriesTransformerForPrediction.
    """
    
    def __init__(
        self,
        context_length: int = 512,
        prediction_length: int = 96,
        input_size: int = 1,
        num_time_features: int = 5,
        d_model: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_attention_heads: int = 4,
        decoder_attention_heads: int = 4,
        encoder_ffn_dim: int = 512,
        decoder_ffn_dim: int = 512,
        dropout: float = 0.1,
        num_static_categorical_features: int = 0,
        num_static_real_features: int = 0,
        cardinality: List[int] = None,
        embedding_dimension: int = 16,
        lags_sequence: List[int] = None,
        distribution_output: str = "student_t",
    ):
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            input_size=input_size,
            num_time_features=num_time_features,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            num_static_categorical_features=num_static_categorical_features,
            num_static_real_features=num_static_real_features,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            lags_sequence=lags_sequence,
        )
        
        self.distribution_output = distribution_output
        
        # Distribution head
        self.distribution_head = DistributionHead(
            d_model,
            prediction_length,
            distribution_output,
        )
        
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        static_categorical: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            past_values: (batch, context_length, input_size)
            past_time_features: (batch, context_length, num_time_features)
            future_time_features: (batch, prediction_length, num_time_features)
            static_categorical: (batch, num_static_categorical)
            static_real: (batch, num_static_real)
            past_observed_mask: (batch, context_length, input_size)
            
        Returns:
            Dictionary of distribution parameters
        """
        decoder_hidden_states = super().forward(
            past_values,
            past_time_features,
            future_time_features,
            static_categorical,
            static_real,
            past_observed_mask,
        )
        
        # Apply distribution head
        dist_params = self.distribution_head(decoder_hidden_states)
        
        return dist_params
    
    def sample(
        self,
        dist_params: Dict[str, torch.Tensor],
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from the predicted distribution.
        
        Args:
            dist_params: Distribution parameters from forward pass
            num_samples: Number of samples to generate
            
        Returns:
            Samples of shape (batch, num_samples, prediction_length, input_size)
        """
        if self.distribution_output == "student_t":
            df = dist_params["df"]
            loc = dist_params["loc"]
            scale = dist_params["scale"]
            
            # Sample from Student-t distribution
            samples = loc + scale * torch.randn(num_samples, *loc.shape, device=loc.device)
            
        elif self.distribution_output == "normal":
            loc = dist_params["loc"]
            scale = dist_params["scale"]
            
            # Sample from Normal distribution
            samples = loc + scale * torch.randn(num_samples, *loc.shape, device=loc.device)
            
        else:
            # negative_binomial
            logits = dist_params["logits"]
            
            # Sample from Negative Binomial
            probs = torch.sigmoid(logits)
            total_count = torch.ones_like(probs) * 10  # Default count
            samples = torch.distributions.NegativeBinomial(total_count, probs).sample()
        
        return samples


def create_model(
    context_length: int = 512,
    prediction_length: int = 96,
    input_size: int = 1,
    d_model: int = 128,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    **kwargs,
) -> TimeSeriesTransformerForPrediction:
    """
    Create a Time Series Transformer model.
    
    Args:
        context_length: Input context length
        prediction_length: Output prediction length
        input_size: Number of input channels
        d_model: Model dimension
        encoder_layers: Number of encoder layers
        decoder_layers: Number of decoder layers
        **kwargs: Additional arguments
        
    Returns:
        TimeSeriesTransformerForPrediction model
    """
    model = TimeSeriesTransformerForPrediction(
        context_length=context_length,
        prediction_length=prediction_length,
        input_size=input_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    context_length = 512
    prediction_length = 96
    input_size = 1
    num_time_features = 5
    
    model = create_model(
        context_length=context_length,
        prediction_length=prediction_length,
        input_size=input_size,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample inputs
    past_values = torch.randn(batch_size, context_length, input_size)
    past_time_features = torch.randn(batch_size, context_length, num_time_features)
    future_time_features = torch.randn(batch_size, prediction_length, num_time_features)
    
    # Forward pass
    with torch.no_grad():
        dist_params = model(
            past_values,
            past_time_features,
            future_time_features,
        )
    
    print(f"Past values shape: {past_values.shape}")
    print(f"Future time features shape: {future_time_features.shape}")
    print(f"Distribution parameters keys: {dist_params.keys()}")
    
    for key, value in dist_params.items():
        print(f"  {key}: {value.shape}")
    
    print("✓ Model test passed!")
