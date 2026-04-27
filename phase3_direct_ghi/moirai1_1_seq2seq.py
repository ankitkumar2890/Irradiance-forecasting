"""Custom seq2seq forecaster built on the Moirai 1.1 backbone.

This module reuses the pretrained Moirai 1.1 input projection + transformer
encoder for the historical context, then adds a separate decoder stack that
consumes future-known covariates to predict the future GHI trajectory.
"""
from __future__ import annotations

import torch
from torch import nn

from uni2ts.model.moirai import MoiraiModule
from uni2ts.model.moirai.module import packed_attention_mask


class Moirai1Seq2Seq(nn.Module):
    """Encoder-decoder forecaster using Moirai 1.1 as the history encoder."""

    def __init__(
        self,
        future_feat_dim: int,
        model_id: str = "Salesforce/moirai-1.1-R-small",
        patch_size: int = 8,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.patch_size = patch_size
        self.future_feat_dim = future_feat_dim
        self.local_files_only = local_files_only

        self.backbone = MoiraiModule.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        if patch_size not in self.backbone.patch_sizes:
            raise ValueError(
                f"patch_size={patch_size} is not supported by {model_id}. "
                f"Available patch sizes: {list(self.backbone.patch_sizes)}"
            )

        d_model = self.backbone.d_model
        self.max_patch_size = max(self.backbone.patch_sizes)
        self.d_model = d_model
        self.future_input_proj = nn.Linear(future_feat_dim * patch_size, d_model)
        self.decoder_pos_emb = nn.Embedding(self.backbone.max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _patchify(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, feat_dim = values.shape
        if time_steps % self.patch_size != 0:
            raise ValueError(
                f"time_steps={time_steps} must be divisible by patch_size={self.patch_size}"
            )
        num_patches = time_steps // self.patch_size
        return values.reshape(batch_size, num_patches, self.patch_size, feat_dim)

    @staticmethod
    def _normalize(
        past_target: torch.Tensor,
        past_dynamic_real: torch.Tensor,
        future_dynamic_real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mean = past_target.mean(dim=1, keepdim=True)
        target_std = past_target.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        past_target_norm = (past_target - target_mean) / target_std

        dyn_mean = past_dynamic_real.mean(dim=1, keepdim=True)
        dyn_std = past_dynamic_real.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        past_dynamic_norm = (past_dynamic_real - dyn_mean) / dyn_std
        future_dynamic_norm = (future_dynamic_real - dyn_mean) / dyn_std
        return (
            past_target_norm,
            past_dynamic_norm,
            future_dynamic_norm,
            target_mean,
            target_std,
        )

    def encode(
        self,
        past_target: torch.Tensor,
        past_dynamic_real: torch.Tensor,
    ) -> torch.Tensor:
        full_past = torch.cat([past_target, past_dynamic_real], dim=-1)
        patched = self._patchify(full_past).permute(0, 3, 1, 2).contiguous()
        batch_size, num_vars, num_patches, _ = patched.shape
        tokens = patched.reshape(batch_size, num_vars * num_patches, self.patch_size)
        if self.patch_size < self.max_patch_size:
            tokens = torch.nn.functional.pad(tokens, (0, self.max_patch_size - self.patch_size))
        patch_sizes = torch.full(
            (batch_size, num_vars * num_patches),
            self.patch_size,
            dtype=torch.long,
            device=past_target.device,
        )
        token_embeddings = self.backbone.in_proj(tokens, patch_sizes)

        device = past_target.device
        time_id = (
            torch.arange(num_patches, device=device)
            .repeat(num_vars)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        var_id = (
            torch.arange(num_vars, device=device)
            .repeat_interleave(num_patches)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        sample_id = torch.zeros(
            (batch_size, num_vars * num_patches),
            dtype=torch.long,
            device=device,
        )
        return self.backbone.encoder(
            token_embeddings,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=var_id,
        )

    def decode(
        self,
        memory: torch.Tensor,
        future_dynamic_real: torch.Tensor,
    ) -> torch.Tensor:
        patched_future = self._patchify(future_dynamic_real)
        batch_size, num_patches, _, feat_dim = patched_future.shape
        decoder_inputs = patched_future.reshape(batch_size, num_patches, self.patch_size * feat_dim)
        decoder_inputs = self.future_input_proj(decoder_inputs)

        positions = torch.arange(num_patches, device=future_dynamic_real.device)
        decoder_inputs = decoder_inputs + self.decoder_pos_emb(positions).unsqueeze(0)

        causal_mask = torch.triu(
            torch.ones(num_patches, num_patches, device=future_dynamic_real.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=causal_mask,
        )
        patch_preds = self.output_head(decoded)
        return patch_preds.reshape(batch_size, num_patches * self.patch_size)

    def forward(
        self,
        past_target: torch.Tensor,
        past_dynamic_real: torch.Tensor,
        future_dynamic_real: torch.Tensor,
    ) -> torch.Tensor:
        (
            past_target_norm,
            past_dynamic_norm,
            future_dynamic_norm,
            target_mean,
            target_std,
        ) = self._normalize(past_target, past_dynamic_real, future_dynamic_real)
        memory = self.encode(past_target_norm, past_dynamic_norm)
        preds_norm = self.decode(memory, future_dynamic_norm).unsqueeze(-1)
        preds = preds_norm * target_std + target_mean
        return preds.squeeze(-1)
