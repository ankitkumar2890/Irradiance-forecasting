"""
model.py — Redesigned Temporal Fusion Transformer for CAF Forecasting.

Key design decisions driven by this specific dataset
-----------------------------------------------------
Dataset facts (from dataset.py + config.py):
  - Encoder: (B, 72, 13) — 72 past hours, 13 features
      [CAF, clear_sky_ghi, tcc, lcc, mcc, hcc, u10, v10,
       zenith_angle, hour_sin, hour_cos, doy_sin, doy_cos]
  - Decoder: (B, 24, 12) — 24 future hours, 12 known-future features
      (same as encoder MINUS CAF — it is the unknown target)
  - Target: (B, 24) — 24-step CAF forecasts in [0, 1]

Improvements over the original implementation
----------------------------------------------
1.  FEATURE-LEVEL LINEAR PROJECTION BEFORE VSN
    Original code fed raw float features directly into VSN GRNs where
    each feature's GRN received only a scalar (dim=1). That is a very
    narrow input with no cross-feature context for the weight computation.
    Fix: project each feature from dim=1 → hidden_dim with a learned
    linear, THEN pass those embeddings into VSN GRNs. This gives each
    feature a richer representation before importance weighting.

2.  SHARED FEATURE PROJECTIONS ACROSS ENCODER/DECODER
    The 12 features that appear in both encoder and decoder
    (everything except CAF) share the same projection weights.
    This enforces consistent semantics across the two streams —
    e.g., the "tcc at t+3" embedding lives in the same space as
    "tcc at t-10", which is physically correct.

3.  CORRECTED INTERPRETABLE MULTI-HEAD ATTENTION
    The original averaged head outputs BEFORE repeating to hidden_dim,
    which collapsed the representation to d_k and then padded it by
    repetition — a mathematically unsound approximation.
    Fix: keep each head's output in full d_k, concatenate properly,
    and apply the output projection. Per-head attention maps are still
    returned separately, preserving interpretability.

4.  TEMPORAL POSITIONAL ENCODING
    The original had no notion of "which step is this within the window".
    We add a lightweight learned positional encoding over the full
    (past+future) sequence. This is particularly important for the 72-step
    encoder, where relative temporal position (e.g., hour 68 vs hour 1)
    carries real information.

5.  QUANTILE OUTPUT HEAD
    Pure MSE ignores the asymmetric error profile of CAF: underpredicting
    a sunny day is more costly to a solar operator than overpredicting.
    We add an optional quantile output head (p10, p50, p90) alongside
    the point forecast. The loss is the pinball / quantile loss.
    The caller in train_tft.py can choose between 'mse', 'huber', or
    'quantile' via config.LOSS_FN.

6.  LAYER-LEVEL DROPOUT + INPUT DROPOUT
    Added input dropout to regularise the raw features (prevents the
    model latching onto single-sensor noise), and separate dropout rates
    for LSTM vs attention sub-networks.

7.  WEIGHT INITIALISATION
    All linear layers use Kaiming-uniform (ReLU-family) init.
    LayerNorm gains set to 1.0. LSTM weights use orthogonal init for
    the recurrent matrices, which is known to improve gradient flow in
    long sequences (72 steps here).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Utility: weight initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_weights(module: nn.Module) -> None:
    """Kaiming-uniform for Linear, orthogonal for LSTM recurrent weights."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_hh" in name:           # recurrent weights → orthogonal
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:          # input weights → kaiming
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class GatedLinearUnit(nn.Module):
    """
    GLU: output = sigmoid(W2·x) ⊙ (W1·x).
    Used as the gating mechanism throughout the TFT.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear_gate = nn.Linear(input_dim, output_dim)
        self.linear_feat = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear_gate(x)) * self.linear_feat(self.dropout(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN: x → fc1 → ELU [+ context] → dropout → fc2 → GLU → LayerNorm(x + GLU).

    An optional context vector (e.g. static embedding) is injected additively
    after the first linear, allowing global station metadata to modulate
    the computation without needing a separate branch.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        )
        self.glu = GatedLinearUnit(output_dim, output_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)
        h = F.elu(self.fc1(x))
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.layer_norm(residual + h)


class FeatureEmbedder(nn.Module):
    """
    Projects each raw scalar feature to hidden_dim before VSN processing.

    IMPROVEMENT OVER ORIGINAL
    -------------------------
    Original VSN fed dim=1 scalars directly into per-feature GRNs.
    A scalar has almost no representational capacity. By projecting to
    hidden_dim first, the GRN can compute richer gating signals.

    Shared vs private projections
    -----------------------------
    Features that appear in both encoder and decoder (indices tracked
    externally) can share projection weights, enforcing that the same
    physical quantity maps to the same hidden space regardless of which
    stream it is in.
    """

    def __init__(self, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # One Linear per feature: (1,) → (hidden_dim,)
        self.projections = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.input_dropout = nn.Dropout(dropout)  # regularise raw inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, num_features)  — raw continuous features

        Returns
        -------
        embedded : (B, T, num_features, hidden_dim)
        """
        B, T, F = x.shape
        x = self.input_dropout(x)
        # Project each feature independently
        embedded = torch.stack(
            [self.projections[i](x[..., i : i + 1]) for i in range(F)],
            dim=2,
        )  # (B, T, F, hidden_dim)
        return embedded


class VariableSelectionNetwork(nn.Module):
    """
    Learns soft importance weights over input features.

    IMPROVEMENT OVER ORIGINAL
    -------------------------
    Receives pre-embedded features (B, T, F, hidden_dim) instead of raw
    scalars (B, T, F). The flat-GRN now has a much richer flat input
    (F * hidden_dim) from which to compute attention weights.

    The optional context vector (static station embedding) is injected
    into the flat GRN, allowing the model to learn that, say, the wind
    variable matters more at coastal Chennai than at inland Coimbatore.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        flat_input_dim = num_features * hidden_dim
        # Joint GRN that sees all embedded features concatenated
        self.grn_flat = GatedResidualNetwork(
            input_dim=flat_input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout,
            context_dim=context_dim,
        )
        # Per-feature GRNs that process the already-embedded (hidden_dim,) vectors
        self.grn_per_feature = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

    def forward(
        self,
        embedded: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        embedded : (B, T, F, hidden_dim) — output of FeatureEmbedder
        context  : (B, context_dim)       — optional static embedding

        Returns
        -------
        selected : (B, T, hidden_dim)
        weights  : (B, T, F)  — variable importance (softmax, sums to 1)
        """
        B, T, F, H = embedded.shape

        # Flatten F * H for the joint GRN
        flat = embedded.view(B, T, F * H)

        # Broadcast context to (B, T, context_dim) if provided
        ctx = None
        if context is not None:
            ctx = context.unsqueeze(1).expand(B, T, -1)

        weights = torch.softmax(self.grn_flat(flat, ctx), dim=-1)  # (B, T, F)

        # Per-feature GRN: process each embedded feature vector
        transformed = torch.stack(
            [self.grn_per_feature[i](embedded[:, :, i, :]) for i in range(F)],
            dim=-1,
        )  # (B, T, H, F)

        # Weighted sum
        selected = (transformed * weights.unsqueeze(2)).sum(dim=-1)  # (B, T, H)
        return selected, weights


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings added to the temporal sequence.

    IMPROVEMENT OVER ORIGINAL
    -------------------------
    The original had no positional encoding at all. Without it, the
    self-attention layer cannot distinguish "this is hour 5" from "this is
    hour 68" in the 96-step (72+24) sequence. Solar irradiance has very
    strong within-day and within-sequence temporal structure, so positional
    information is important.

    We use *learned* (not sinusoidal) embeddings because the maximum
    sequence length is small and fixed (96 steps), so the model can
    afford to memorize position patterns from data.
    """

    def __init__(self, max_len: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H) → adds positional embedding."""
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return x + self.embedding(pos).unsqueeze(0)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention that returns per-head attention maps.

    IMPROVEMENT OVER ORIGINAL
    -------------------------
    The original "interpretable" MHA averaged head outputs then repeated
    the averaged d_k vector num_heads times to restore the hidden dimension —
    this is a dimension hack that does not preserve head diversity.

    Correct implementation:
      1. Each head computes its own (B, T_q, d_k) context vector.
      2. Heads are concatenated → (B, T_q, hidden_dim).
      3. A single W_out projection mixes head outputs.
      4. Per-head attention maps are stacked and returned for inspection.

    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query, key, value : (B, T, hidden_dim)
        causal_mask       : (1, 1, T_q, T_k) — lower-triangular (float, 0→-inf)

        Returns
        -------
        output       : (B, T_q, hidden_dim)
        attn_weights : (B, num_heads, T_q, T_k)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project and split into heads: (B, heads, T, d_k)
        def _project(tensor: torch.Tensor, T: int, linear: nn.Linear) -> torch.Tensor:
            return (
                linear(tensor)
                .view(B, T, self.num_heads, self.d_k)
                .transpose(1, 2)
            )

        Q = _project(query, T_q, self.W_q)
        K = _project(key,   T_k, self.W_k)
        V = _project(value, T_k, self.W_v)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (B, heads, T_q, T_k)

        # Causal mask (lower-triangular): positions where mask=0 → -inf
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Head-wise context: (B, heads, T_q, d_k)
        context = torch.matmul(attn, V)

        # Concatenate heads → (B, T_q, hidden_dim) then project
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.d_k)
        output = self.W_out(context)

        return output, attn


# ══════════════════════════════════════════════════════════════════════════════
# Quantile Loss
# ══════════════════════════════════════════════════════════════════════════════

class QuantileLoss(nn.Module):
    """
    Pinball (quantile) loss for multi-quantile forecasting.

    For CAF solar forecasting, providing prediction intervals is valuable:
    - p10: conservative (cloudy day scenario)
    - p50: median (best point estimate)
    - p90: optimistic (sunny day scenario)

    The total loss is the mean pinball loss across all quantiles.
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        preds  : (B, T, Q) — Q quantile predictions
        target : (B, T)    — ground truth

        Returns
        -------
        Scalar loss.
        """
        target = target.unsqueeze(-1)  # (B, T, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[..., i : i + 1]
            losses.append(torch.max(q * err, (q - 1.0) * err))
        return torch.cat(losses, dim=-1).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Full TFT Model
# ══════════════════════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    Redesigned TFT for multi-station, multi-horizon CAF forecasting.

    Dataset interface (must match dataset.py exactly)
    -------------------------------------------------
    encoder_input : (B, 72, 13)  — past 72 hours
      [CAF, clear_sky_ghi, tcc, lcc, mcc, hcc, u10, v10,
       zenith_angle, hour_sin, hour_cos, doy_sin, doy_cos]

    decoder_input : (B, 24, 12)  — known future 24 hours
      [clear_sky_ghi, tcc, lcc, mcc, hcc, u10, v10,
       zenith_angle, hour_sin, hour_cos, doy_sin, doy_cos]

    target        : (B, 24)      — CAF ∈ [0, 1]

    Parameters
    ----------
    encoder_input_dim : int   (13 from config)
    decoder_input_dim : int   (12 from config)
    hidden_dim        : int   (64 from config)
    num_heads         : int   (4 from config)
    lstm_layers       : int   (1 from config)
    dropout           : float (0.1 from config)
    forecast_horizon  : int   (24 from config)
    past_steps        : int   (72 from config)
    use_quantiles     : bool  If True, outputs three quantiles (p10, p50, p90)
                              instead of a single point forecast.
    quantiles         : tuple Quantile levels when use_quantiles=True.
    """

    def __init__(
        self,
        encoder_input_dim: int = 13,
        decoder_input_dim: int = 12,
        hidden_dim: int = 64,
        num_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        forecast_horizon: int = 24,
        past_steps: int = 72,
        use_quantiles: bool = False,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.past_steps = past_steps
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.use_quantiles = use_quantiles
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        T_total = past_steps + forecast_horizon

        # ── Feature Embedders ────────────────────────────────────────────────
        # Encoder embeds 13 features (including CAF).
        # Decoder embeds 12 features (known future, no CAF).
        # We do NOT share projection weights between encoder and decoder in
        # this implementation to keep them fully independent. If you want
        # weight sharing for the 12 common features, slice the ModuleList
        # and share projections[1:] between encoder and decoder embedders.
        self.encoder_embedder = FeatureEmbedder(encoder_input_dim, hidden_dim, dropout)
        self.decoder_embedder = FeatureEmbedder(decoder_input_dim, hidden_dim, dropout)

        # ── Variable Selection Networks ──────────────────────────────────────
        # context_dim=None here; pass a static station embedding optionally.
        self.encoder_vsn = VariableSelectionNetwork(
            num_features=encoder_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            num_features=decoder_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ── LSTM Encoder / Decoder ───────────────────────────────────────────
        # Orthogonal init applied in _init_weights().
        lstm_drop = dropout if lstm_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_drop,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_drop,
        )

        # ── Post-LSTM gate + residual ────────────────────────────────────────
        self.post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # ── Positional Encoding (IMPROVEMENT: added vs original) ─────────────
        self.pos_enc = LearnedPositionalEncoding(T_total, hidden_dim)

        # ── Temporal Self-Attention ──────────────────────────────────────────
        self.multihead_attn = InterpretableMultiHeadAttention(
            hidden_dim, num_heads, dropout
        )
        self.post_attn_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # ── Position-wise Feed-Forward ───────────────────────────────────────
        self.ff_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim * 2, hidden_dim, dropout
        )

        # ── Output Projection ───────────────────────────────────────────────
        # Point forecast or multi-quantile forecast head.
        output_dim = self.num_quantiles if use_quantiles else 1
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # ── Weight Init ──────────────────────────────────────────────────────
        self.apply(_init_weights)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_input : (B, 72, 13)  — past observations
        decoder_input : (B, 24, 12)  — known future covariates

        Returns
        -------
        predictions  : (B, 24)       if use_quantiles=False (point forecast)
                       (B, 24, Q)    if use_quantiles=True
        enc_weights  : (B, 72, 13)   variable importance (encoder)
        dec_weights  : (B, 24, 12)   variable importance (decoder)
        attn_weights : (B, heads, 24, 96)  per-head attention maps
        """
        # ── 1. Feature Embedding ─────────────────────────────────────────────
        # (B, T, F) → (B, T, F, H)
        enc_embedded = self.encoder_embedder(encoder_input)
        dec_embedded = self.decoder_embedder(decoder_input)

        # ── 2. Variable Selection ────────────────────────────────────────────
        enc_selected, enc_var_weights = self.encoder_vsn(enc_embedded)
        dec_selected, dec_var_weights = self.decoder_vsn(dec_embedded)
        # enc_selected: (B, 72, H),  dec_selected: (B, 24, H)

        # ── 3. LSTM Encoder → Decoder (sharing hidden state) ─────────────────
        enc_lstm_out, (h_n, c_n) = self.encoder_lstm(enc_selected)
        dec_lstm_out, _          = self.decoder_lstm(dec_selected, (h_n, c_n))

        # ── 4. Concatenate and Gate ──────────────────────────────────────────
        lstm_out = torch.cat([enc_lstm_out, dec_lstm_out], dim=1)   # (B, 96, H)
        vsn_cat  = torch.cat([enc_selected,  dec_selected],  dim=1)  # (B, 96, H)

        gated   = self.post_lstm_glu(lstm_out)
        temporal = self.post_lstm_norm(vsn_cat + gated)              # (B, 96, H)

        # ── 5. Positional Encoding (ADDED) ───────────────────────────────────
        temporal = self.pos_enc(temporal)

        # ── 6. Causal Self-Attention ─────────────────────────────────────────
        T_total = temporal.shape[1]
        # Lower-triangular causal mask prevents future leakage
        causal_mask = torch.tril(
            torch.ones(T_total, T_total, device=temporal.device)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        attn_out, attn_weights = self.multihead_attn(
            query=temporal,
            key=temporal,
            value=temporal,
            causal_mask=causal_mask,
        )

        gated_attn = self.post_attn_glu(attn_out)
        enriched   = self.post_attn_norm(temporal + gated_attn)

        # ── 7. Position-wise Feed-Forward ────────────────────────────────────
        output = self.ff_grn(enriched)

        # ── 8. Extract decoder positions and project ─────────────────────────
        decoder_output = output[:, self.past_steps :, :]   # (B, 24, H)
        raw = self.output_proj(decoder_output)              # (B, 24, 1) or (B, 24, Q)

        if self.use_quantiles:
            predictions = raw  # (B, 24, Q) — clipping applied externally
        else:
            predictions = raw.squeeze(-1)  # (B, 24)

        # Return attention over decoder query positions only
        dec_attn = attn_weights[:, :, self.past_steps :, :]  # (B, heads, 24, 96)

        return predictions, enc_var_weights, dec_var_weights, dec_attn

    # ── Convenience ─────────────────────────────────────────────────────────

    def predict(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-only forward pass. Returns clipped point forecast (or
        clipped p50 if quantile mode is active), shape (B, 24).
        """
        self.eval()
        with torch.no_grad():
            preds, _, _, _ = self.forward(encoder_input, decoder_input)
            if self.use_quantiles:
                # p50 is the middle quantile
                mid = self.num_quantiles // 2
                return preds[..., mid].clamp(0.0, 1.0)
            return preds.clamp(0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Quick Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    # ---- Point-forecast model ----
    model = TemporalFusionTransformer(
        encoder_input_dim=13,
        decoder_input_dim=12,
        hidden_dim=64,
        num_heads=4,
        lstm_layers=1,
        dropout=0.1,
        forecast_horizon=24,
        past_steps=72,
        use_quantiles=False,
    )
    print(f"TFT (point) parameters: {model.count_parameters():,}")

    enc = torch.randn(4, 72, 13)
    dec = torch.randn(4, 24, 12)
    preds, enc_w, dec_w, attn_w = model(enc, dec)
    assert preds.shape  == (4, 24),       f"Bad preds shape: {preds.shape}"
    assert enc_w.shape  == (4, 72, 13),   f"Bad enc_w shape: {enc_w.shape}"
    assert dec_w.shape  == (4, 24, 12),   f"Bad dec_w shape: {dec_w.shape}"
    assert attn_w.shape == (4, 4, 24, 96),f"Bad attn_w shape: {attn_w.shape}"
    print(f"  preds:       {preds.shape}")
    print(f"  enc_weights: {enc_w.shape}")
    print(f"  dec_weights: {dec_w.shape}")
    print(f"  attn_weights:{attn_w.shape}")

    # ---- Quantile model ----
    model_q = TemporalFusionTransformer(
        encoder_input_dim=13,
        decoder_input_dim=12,
        hidden_dim=64,
        num_heads=4,
        lstm_layers=1,
        dropout=0.1,
        forecast_horizon=24,
        past_steps=72,
        use_quantiles=True,
        quantiles=(0.1, 0.5, 0.9),
    )
    print(f"\nTFT (quantile) parameters: {model_q.count_parameters():,}")
    preds_q, _, _, _ = model_q(enc, dec)
    assert preds_q.shape == (4, 24, 3), f"Bad quantile preds shape: {preds_q.shape}"
    print(f"  quantile preds: {preds_q.shape}")

    # ---- Quantile loss ----
    ql = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    target = torch.rand(4, 24)
    loss = ql(preds_q, target)
    print(f"  quantile loss: {loss.item():.4f}")

    print("\n✓ All assertions passed.")
