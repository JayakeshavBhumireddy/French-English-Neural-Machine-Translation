"""
model/embeddings.py
-------------------
Token + positional embeddings for the encoder and decoder.

Improvements over the original:
  1. **Embedding scale** — multiply by sqrt(d_model) as in Vaswani et al.
     This prevents the positional signal from dominating the token signal.
  2. **Separate encoder / decoder** — when using a shared vocabulary we still
     want two separate Embedding tables for encoder and decoder inputs.
     The output projection (lm_head) can then be tied to the decoder
     embedding matrix (weight tying) which cuts parameters significantly.
  3. **Sinusoidal PE as a buffer** — not a Parameter, so it is never saved
     in the checkpoint unnecessarily, and is always recomputed correctly.
  4. **Learned PE option** — an ``nn.Embedding`` variant for experiments.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from fr2en.configs.config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) sin/cos positional encodings from Vaswani et al.
    Registered as a buffer so it moves to the correct device with .to(device)
    but is never saved in the state_dict unnecessarily.
    """

    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)            # (L, 1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10_000.0) / embed_dim)
        )                                                                        # (D/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Shape: (1, max_len, embed_dim) — broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable position embeddings (as in BERT, RoBERTa).
    Usually slightly better than sinusoidal for shorter sequences.
    """

    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return x + self.pe(positions)


class TranslationEmbeddings(nn.Module):
    """
    Token embeddings + positional encodings for both encoder and decoder.

    When ``tie_weights=True`` the src and tgt embedding tables are the same
    object (shared vocabulary assumption).  The lm_head weight is tied
    externally in ``Transformer.__init__``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        D = config.embedding_dim
        V = config.vocab_size
        self._scale = math.sqrt(D)

        # Embedding tables
        self.src_embed = nn.Embedding(V, D, padding_idx=config.pad_token_id)
        if config.tie_weights:
            # Single shared table — encoder and decoder share weights
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(V, D, padding_idx=config.pad_token_id)

        # Positional encodings
        # When use_rope=True, RoPE is applied inside MultiHeadAttention — no PE here.
        if getattr(config, 'use_rope', False):
            self.src_pe = None
            self.tgt_pe = None
        else:
            PECls = LearnedPositionalEncoding if config.learned_pos_embed else SinusoidalPositionalEncoding
            self.src_pe = PECls(config.max_src_len, D)
            self.tgt_pe = PECls(config.max_tgt_len, D)

        self.dropout = nn.Dropout(config.dropout)

        # Initialise token embeddings
        nn.init.normal_(self.src_embed.weight, std=0.02)
        if not config.tie_weights:
            nn.init.normal_(self.tgt_embed.weight, std=0.02)
        # Zero out the padding embedding
        with torch.no_grad():
            self.src_embed.weight[config.pad_token_id].zero_()
            if not config.tie_weights:
                self.tgt_embed.weight[config.pad_token_id].zero_()

    def encode_src(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(input_ids) * self._scale
        if self.src_pe is not None:
            x = self.src_pe(x)
        return self.dropout(x)

    def encode_tgt(self, input_ids: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        """(B, T) -> (B, T, D). step_offset shifts the PE for KV-cache decode_step."""
        x = self.tgt_embed(input_ids) * self._scale
        if self.tgt_pe is not None:
            # SinusoidalPositionalEncoding stores its table as a torch.Tensor buffer.
            # LearnedPositionalEncoding stores it as nn.Embedding — cannot be sliced
            # the same way.  Use isinstance to tell them apart.
            pe_buf = getattr(self.tgt_pe, "pe", None)
            if step_offset > 0 and isinstance(pe_buf, torch.Tensor):
                T = input_ids.size(1)
                x = x + pe_buf[:, step_offset:step_offset + T]
            else:
                x = self.tgt_pe(x)
        return self.dropout(x)
