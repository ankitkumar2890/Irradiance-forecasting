# ==============================================================================
# model_architecture.py — Phase 1: CloudMapper v6 (Residual MLP)
#
# Architecture: Residual MLP with gated ERA5 skip connection.
#
# Design decisions based on the ERA5→ICON problem:
#
#   1. No sequence model (GRU/LSTM): temporal info enters as t-1 lag features.
#      GRU overfits when targets have ~0.5-0.6 correlation ceiling.
#
#   2. 64-dim hidden: with 15 input features and limited target correlation,
#      64 dims is sufficient. 128 risks memorizing noise without improving signal.
#
#   3. Dropout 0.25: higher than typical (0.1-0.15) because targets are
#      fundamentally noisy. Forces the network to learn robust features.
#
#   4. Residual blocks: learn f(x) + x, so the network starts at identity
#      and learns corrections. Combined with α-blend over ERA5 raw, the
#      entire model is a correction machine.
#
#   5. ~21K parameters total — minimal capacity for a low-SNR task.
# ==============================================================================
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → + skip.

    Why pre-norm: LayerNorm before the transformation is more stable than
    post-norm (used in original ResNet). Standard in modern architectures.

    Why skip connection: The block output is x + f(x). This means:
    - If f(x) ≈ 0 (early training), the block is near-identity → stable
    - The network only needs to learn the *residual* correction
    - Gradients flow directly through the skip → no vanishing gradient

    The last linear layer is initialised at 10% scale so f(x) starts near zero.
    """

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.block:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Scale down last linear so block starts near-identity
        last_linear = [m for m in self.block if isinstance(m, nn.Linear)][-1]
        last_linear.weight.data *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TAFResNet(nn.Module):
    """
    Residual MLP for ERA5 → ICON cloud cover mapping.

    Maps 4 ERA5 cloud fractions (total, low, mid, high) → 1 ICON cloud cover.

    Forward pass:
        1. LayerNorm input → project to hidden_dim → GELU → Dropout
        2. Pass through N residual blocks (each: LN → Linear → GELU → Drop → Linear + skip)
        3. LayerNorm → Linear → Sigmoid → head_out ∈ [0, 1]
        4. α-blend: output = α × head_out + (1-α) × ERA5_total_cloud_cover
        5. Clamp to [0, 1]

    The α-blend uses ERA5 total_cloud_cover as the residual baseline
    (closest analog to ICON cloud_cover). The network learns corrections.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_res_blocks: int = 2,
        output_size: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self._frozen = False

        # Input projection: 15 features → 64-dim hidden space
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout)

        # Residual backbone: 2 blocks of (LN → Linear → GELU → Drop → Linear + skip)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Output head: LN → Linear → Sigmoid
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid(),
        )

        # Learned α for the residual blend.
        # Stored as logit so unconstrained optimisation maps to [0,1] via sigmoid.
        # logit(0.6) ≈ 0.405 → network starts slightly trusting itself.
        self.alpha_logit = nn.Parameter(torch.tensor(0.405))

        self._init_head()

    def _init_head(self):
        """Xavier init for head, Kaiming for input projection."""
        for m in self.output_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        era5_raw: "torch.Tensor" = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch, input_size) — StandardScaler'd tabular features
            era5_raw: (batch, 1) — UNSCALED ERA5 total_cloud_cover [0,1]
                      for the residual skip. If None, returns pure network output.

        Returns:
            (batch, 1) — predicted ICON cloud_cover ∈ [0, 1]
        """
        # Project input to hidden space
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = self.input_act(h)
        h = self.input_drop(h)

        # Residual backbone
        h = self.res_blocks(h)

        # Head → sigmoid
        head_out = self.output_head(h)  # (batch, 1) ∈ [0, 1]

        # Gated residual blend with ERA5 total_cloud_cover
        if era5_raw is not None:
            alpha = torch.sigmoid(self.alpha_logit)  # scalar ∈ [0, 1]
            output = alpha * head_out + (1.0 - alpha) * era5_raw
            return output.clamp(0.0, 1.0)

        return head_out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def freeze(self):
        """Freeze all parameters for inference."""
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get_alpha_values(self) -> dict:
        """Return the learned α blending weight."""
        alpha = float(torch.sigmoid(self.alpha_logit).detach().cpu())
        return {"cloud_cover": alpha}