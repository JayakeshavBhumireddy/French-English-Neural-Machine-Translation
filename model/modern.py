"""
model/modern.py
---------------
Modern architecture components that replace the originals in layers.py
and embeddings.py when config flags are set.

SwiGLU FFN  — used by LLaMA, PaLM, Gemini, Mistral
RoPE        — used by LLaMA, Mistral, Qwen, GPT-NeoX

Both are drop-in replacements: same input/output shapes, new internals.
Enable via config:
    config.model.use_swiglu = True
    config.model.use_rope   = True
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fr2en.configs.config import ModelConfig


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU gated FFN (Shazeer 2020, used in LLaMA/PaLM/Gemini).

    Formula:
        FFN(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down

    Uses 3 weight matrices instead of 2, but ffn_dim is set to
    ⅔ × (4 × D) so parameter count stays the same as standard FFN.

    Standard FFN:  2 × D × 4D = 8D²
    SwiGLU FFN:    3 × D × (8D/3) ≈ 3 × D × 2.67D = 8D²  ✓
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        D = config.embedding_dim
        # ⅔ × 4D, rounded to nearest multiple of 64 for hardware alignment
        hidden = int(2 * config.ffn_dim / 3)
        hidden = (hidden + 63) // 64 * 64

        self.w_gate = nn.Linear(D, hidden, bias=False)
        self.w_up   = nn.Linear(D, hidden, bias=False)
        self.w_down = nn.Linear(hidden, D, bias=False)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # SiLU ≡ x * sigmoid(x)
        up   = self.w_up(x)
        return self.drop(self.w_down(gate * up))


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al. 2021, used by LLaMA/Mistral/Qwen).

    Instead of adding a positional vector to token embeddings, RoPE rotates
    the Q and K vectors by position-dependent angles. This means:
      - Position information is encoded in the *relative* angle between Q and K
      - Absolute position is never baked into the representations
      - Length generalisation is significantly better than sinusoidal PE

    Usage:
        rope = RotaryEmbedding(head_dim, max_seq_len)
        q, k = rope(q, k)              # training: offset=0 (default)
        q, k = rope(q, k, offset=t)   # KV-cache: token is at position t

    The embeddings are buffers (not parameters) — they live on whatever
    device the module is moved to with .to(device).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10_000) -> None:
        super().__init__()
        self.head_dim = head_dim
        # Frequencies: θ_i = 1 / 10000^(2i/d), shape (D/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        # Pre-compute cos/sin cache for max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (L, D/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (L, D)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])  # (1,1,L,D)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])  # (1,1,L,D)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dim to the front, negated."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,    # (B, H, T, Dh)
        k: torch.Tensor,    # (B, H, T, Dh)
        offset: int = 0,    # position of the first token in q/k
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to q and k starting at absolute position `offset`.

        offset=0  → full sequence (training / first decode step)
        offset=t  → KV-cache step: q and k are the single new token at position t
        """
        T   = q.size(2)
        end = offset + T

        if end > self.cos_cache.size(2):
            self._build_cache(end)

        cos = self.cos_cache[:, :, offset:end, :].to(q.dtype)  # (1,1,T,Dh)
        sin = self.sin_cache[:, :, offset:end, :].to(q.dtype)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


