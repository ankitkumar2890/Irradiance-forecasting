"""
TFT Model — Full Temporal Fusion Transformer in pure PyTorch.

Architecture follows Bryan Lim et al., "Temporal Fusion Transformers for
Interpretable Multi-Horizon Time Series Forecasting" (2019).

Components
----------
1. GatedResidualNetwork (GRN) — non-linear gated processing block
2. VariableSelectionNetwork (VSN) — learns per-feature importance weights
3. GatedLinearUnit (GLU) — gating mechanism
4. InterpretableMultiHeadAttention — produces per-head attention weights
5. TemporalFusionTransformer — full model combining all pieces
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class GatedLinearUnit(nn.Module):
    """GLU activation: element-wise gating."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc2(x)) * self.fc1(x)


class GatedResidualNetwork(nn.Module):
    """
    GRN with optional context vector.

    x → Linear → ELU → Linear → GLU → LayerNorm(residual + GLU)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ
        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else None
        )

    def forward(self, x, context=None):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.layer_norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """
    Learns soft weights over input features using per-variable GRNs
    and a joint softmax gate.

    Input:  (batch, time, num_features, feature_dim_each=1)
    Output: (batch, time, hidden_dim)  — weighted sum of transformed features
    """

    def __init__(self, input_dim: int, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features

        # Flatten all features → joint GRN → softmax weights
        self.grn_flat = GatedResidualNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=num_features, dropout=dropout,
        )

        # Per-feature GRN: project single feature → hidden_dim
        self.grn_per_feature = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, time, num_features)

        Returns
        -------
        selected : Tensor, shape (batch, time, hidden_dim)
        weights  : Tensor, shape (batch, time, num_features) — variable importance
        """
        # Variable selection weights
        weights = torch.softmax(self.grn_flat(x), dim=-1)  # (B, T, num_features)

        # Transform each feature independently
        transformed = []
        for i in range(self.num_features):
            feat_i = x[..., i : i + 1]  # (B, T, 1)
            transformed.append(self.grn_per_feature[i](feat_i))  # (B, T, H)
        transformed = torch.stack(transformed, dim=-1)  # (B, T, H, num_features)

        # Weighted combination
        weights_expanded = weights.unsqueeze(2)  # (B, T, 1, num_features)
        selected = (transformed * weights_expanded).sum(dim=-1)  # (B, T, H)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with per-head interpretable weights.

    Unlike standard MHA, each head produces its own attention weights and
    the final output is a *learned weighted average* of head outputs (not
    concatenation), preserving head-level interpretability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project
        Q = self.W_q(query).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, heads, T_q, d_k)

        # Interpretable: average across heads, then project
        context = context.mean(dim=1)  # (B, T_q, d_k)
        # Pad back to hidden_dim
        context = context.repeat(1, 1, self.num_heads)  # (B, T_q, hidden_dim)
        output = self.W_out(context)

        return output, attn


# ══════════════════════════════════════════════════════════════════════════════
# Full TFT Model
# ══════════════════════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    Complete TFT for multi-horizon point forecasting.

    Parameters
    ----------
    encoder_input_dim : int
        Number of features in past/encoder input (8: CAF + 7 covariates).
    decoder_input_dim : int
        Number of known-future features in decoder input (7 covariates).
    hidden_dim : int
        Hidden size across all sub-networks.
    num_heads : int
        Number of attention heads.
    lstm_layers : int
        Number of LSTM layers in encoder/decoder.
    dropout : float
        Dropout rate.
    forecast_horizon : int
        Number of future steps to predict.
    """

    def __init__(
        self,
        encoder_input_dim: int = 8,
        decoder_input_dim: int = 7,
        hidden_dim: int = 64,
        num_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        forecast_horizon: int = 24,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim

        # ── Variable Selection Networks ──────────────────────────────────────
        self.encoder_vsn = VariableSelectionNetwork(
            input_dim=encoder_input_dim,
            num_features=encoder_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            input_dim=decoder_input_dim,
            num_features=decoder_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ── Locality: LSTM Encoder / Decoder ─────────────────────────────────
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Post-LSTM gate + norm ────────────────────────────────────────────
        self.post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # ── Temporal Self-Attention ──────────────────────────────────────────
        self.multihead_attn = InterpretableMultiHeadAttention(
            hidden_dim, num_heads, dropout
        )
        self.post_attn_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # ── Position-wise Feed-Forward ───────────────────────────────────────
        self.ff_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        # ── Output Projection ───────────────────────────────────────────────
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_input, decoder_input):
        """
        Parameters
        ----------
        encoder_input : (batch, past_steps, encoder_input_dim)
        decoder_input : (batch, future_steps, decoder_input_dim)

        Returns
        -------
        predictions : (batch, future_steps)
        enc_weights : (batch, past_steps, encoder_input_dim)   — variable importance
        dec_weights : (batch, future_steps, decoder_input_dim) — variable importance
        attn_weights: (batch, num_heads, future_steps, past+future_steps)
        """
        # ── 1. Variable Selection ────────────────────────────────────────────
        enc_selected, enc_var_weights = self.encoder_vsn(encoder_input)  # (B, T_enc, H)
        dec_selected, dec_var_weights = self.decoder_vsn(decoder_input)  # (B, T_dec, H)

        # ── 2. LSTM Encoder → Decoder ────────────────────────────────────────
        enc_lstm_out, (h_n, c_n) = self.encoder_lstm(enc_selected)
        dec_lstm_out, _ = self.decoder_lstm(dec_selected, (h_n, c_n))

        # Concatenate encoder + decoder temporal outputs
        lstm_out = torch.cat([enc_lstm_out, dec_lstm_out], dim=1)  # (B, T_enc+T_dec, H)

        # Post-LSTM gating + residual
        vsn_cat = torch.cat([enc_selected, dec_selected], dim=1)
        gated = self.post_lstm_glu(lstm_out)
        temporal = self.post_lstm_norm(vsn_cat + gated)  # (B, T_total, H)

        # ── 3. Temporal Self-Attention ───────────────────────────────────────
        # Causal mask: future positions can only attend to past + current
        T_total = temporal.shape[1]
        causal_mask = torch.tril(torch.ones(T_total, T_total, device=temporal.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        attn_out, attn_weights = self.multihead_attn(
            temporal, temporal, temporal, mask=causal_mask
        )

        # Post-attention gating + residual
        gated_attn = self.post_attn_glu(attn_out)
        enriched = self.post_attn_norm(temporal + gated_attn)  # (B, T_total, H)

        # ── 4. Position-wise Feed-Forward ────────────────────────────────────
        output = self.ff_grn(enriched)  # (B, T_total, H)

        # ── 5. Output: take only decoder positions ───────────────────────────
        T_enc = encoder_input.shape[1]
        decoder_output = output[:, T_enc:, :]  # (B, T_dec, H)
        predictions = self.output_proj(decoder_output).squeeze(-1)  # (B, T_dec)

        return predictions, enc_var_weights, dec_var_weights, attn_weights

    def predict(self, encoder_input, decoder_input):
        """Convenience method: returns only the point forecast, clipped to [0, 1]."""
        self.eval()
        with torch.no_grad():
            preds, _, _, _ = self.forward(encoder_input, decoder_input)
        return preds.clamp(0.0, 1.0)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = TemporalFusionTransformer(
        encoder_input_dim=8,
        decoder_input_dim=7,
        hidden_dim=64,
        num_heads=4,
        lstm_layers=1,
        dropout=0.1,
        forecast_horizon=24,
    )
    print(f"TFT parameters: {model.count_parameters():,}")

    # Dummy forward pass
    enc = torch.randn(4, 72, 8)
    dec = torch.randn(4, 24, 7)
    preds, enc_w, dec_w, attn_w = model(enc, dec)
    print(f"Predictions: {preds.shape}")
    print(f"Encoder variable weights: {enc_w.shape}")
    print(f"Decoder variable weights: {dec_w.shape}")
    print(f"Attention weights: {attn_w.shape}")
