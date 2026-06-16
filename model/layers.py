"""
model/layers.py  (v3)
---------------------
EncoderLayer / DecoderLayer updated to:
  - Unpack (output, _) from MultiHeadAttention (which now returns present_kv too)
  - Thread KV-cache through decoder during inference
  - SwiGLU / gradient checkpointing unchanged from v2
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from fr2en.configs.config import ModelConfig
from fr2en.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.use_swiglu:
            from fr2en.model.modern import SwiGLUFeedForward
            self.net = SwiGLUFeedForward(config)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.embedding_dim, config.ffn_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ffn_dim, config.embedding_dim),
                nn.Dropout(config.dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.self_attn      = MultiHeadAttention(config)
        self.ff             = FeedForward(config)
        self.norm1          = nn.LayerNorm(config.embedding_dim)
        self.norm2          = nn.LayerNorm(config.embedding_dim)
        self.dropout        = nn.Dropout(config.dropout)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor, src_mask) -> torch.Tensor:
        attn_out, _ = self.self_attn(self.norm1(x), padding_mask=src_mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor, src_mask=None) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return ckpt.checkpoint(self._forward, x, src_mask, use_reentrant=False)
        return self._forward(x, src_mask)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.self_attn      = MultiHeadAttention(config)
        self.cross_attn     = MultiHeadAttention(config)
        self.ff             = FeedForward(config)
        self.norm1          = nn.LayerNorm(config.embedding_dim)
        self.norm2          = nn.LayerNorm(config.embedding_dim)
        self.norm3          = nn.LayerNorm(config.embedding_dim)
        self.dropout        = nn.Dropout(config.dropout)
        self.use_checkpoint = use_checkpoint

    def _forward(self, tgt, enc_out, src_mask, tgt_mask, past_self_kv=None):
        # Causal self-attention (with optional KV-cache during inference)
        self_out, present_self_kv = self.self_attn(
            self.norm1(tgt), padding_mask=tgt_mask, causal=True, past_kv=past_self_kv
        )
        tgt = tgt + self.dropout(self_out)
        # Cross-attention — keys/values always from encoder, no cache needed
        cross_out, _ = self.cross_attn(
            enc_out, tgt=self.norm2(tgt), padding_mask=src_mask
        )
        tgt = tgt + self.dropout(cross_out)
        tgt = tgt + self.ff(self.norm3(tgt))
        return tgt, present_self_kv

    def forward(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
        past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.use_checkpoint and self.training:
            # Gradient checkpointing doesn't support non-tensor args (past_kv None is fine at train time)
            out = ckpt.checkpoint(
                self._forward, tgt, enc_out, src_mask, tgt_mask, None, use_reentrant=False
            )
            return out
        return self._forward(tgt, enc_out, src_mask, tgt_mask, past_self_kv)
