"""
model/attention.py  (v2)
------------------------
Multi-head attention with optional:
  - RoPE positional encoding (config.use_rope)
  - Grouped Query Attention / Multi-Query Attention (config.num_kv_heads < num_heads)
  - KV-cache for fast autoregressive decoding (past_kv argument)

Baseline behaviour (use_rope=False, num_kv_heads==num_heads, no past_kv)
is identical to v1 — all existing tests still pass.

KV-cache protocol
-----------------
  past_kv : Optional[Tuple[Tensor, Tensor]]
      The cached (key, value) tensors from the previous decode step,
      each of shape (B, num_kv_heads, T_past, head_dim).
  Returns: (output, present_kv)
      present_kv is the new (key, value) to cache for the next step.
      Shape: (B, num_kv_heads, T_past + 1, head_dim).

When past_kv is None the module behaves exactly as before (training, full
sequence encoding, first decode step).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fr2en.configs.config import ModelConfig


class MultiHeadAttention(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0
        self.embed_dim   = config.embedding_dim
        self.num_heads   = config.num_heads
        self.num_kv_heads = config.num_kv_heads   # == num_heads unless GQA/MQA
        self.head_dim    = config.embedding_dim // config.num_heads
        self.attn_drop   = config.attention_dropout
        self.use_rope    = config.use_rope

        # Q projection always outputs num_heads × head_dim
        self.q_proj = nn.Linear(config.embedding_dim, self.num_heads * self.head_dim, bias=False)
        # K/V projections output num_kv_heads × head_dim (smaller when GQA/MQA)
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.embedding_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.embedding_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        # RoPE — built lazily on first forward so head_dim is known
        self._rope: Optional[nn.Module] = None
        if self.use_rope:
            from fr2en.model.modern import RotaryEmbedding
            max_len = max(config.max_src_len, config.max_tgt_len)
            self._rope = RotaryEmbedding(self.head_dim, max_seq_len=max_len)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H*Dh) → (B, H, T, Dh)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reshape_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, Hkv*Dh) → (B, Hkv, T, Dh)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """GQA: repeat each KV head n_rep times → (B, H, T, Dh)."""
        if n_rep == 1:
            return x
        B, Hkv, T, D = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, Hkv, n_rep, T, D)
            .reshape(B, Hkv * n_rep, T, D)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parameters
        ----------
        src          : (B, S, D)
        tgt          : (B, T, D) — if None, self-attention on src
        padding_mask : (B, S) bool, True = real token
        causal       : apply causal mask (decoder self-attn, training)
        past_kv      : cached (K, V) from previous decode step

        Returns
        -------
        output   : (B, T, D)
        present_kv : updated (K, V) cache tuple, or None when past_kv not used
        """
        if tgt is None:
            q_in = k_in = v_in = src
            T = src.size(1)
        else:
            q_in, k_in, v_in = tgt, src, src
            T = tgt.size(1)

        q = self._reshape_q(self.q_proj(q_in))   # (B, H,   T,    Dh)
        k = self._reshape_kv(self.k_proj(k_in))  # (B, Hkv, S,    Dh)
        v = self._reshape_kv(self.v_proj(v_in))  # (B, Hkv, S,    Dh)

        # Apply RoPE to Q and K (self-attention only; cross-attention skips).
        # offset = number of tokens already in the KV cache so the new token
        # gets the right absolute position embedding, not position 0 again.
        if self.use_rope and self._rope is not None and tgt is None:
            offset = past_kv[0].size(2) if past_kv is not None else 0
            q, k = self._rope(q, k, offset=offset)

        # Prepend cached K/V when decoding step-by-step
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        # Always return present_kv for self-attention (enables KV-cache build-up)
        present_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = (k, v) if tgt is None else None

        # GQA: expand KV heads to match Q heads for SDPA
        n_rep = self.num_heads // self.num_kv_heads
        k = self._expand_kv(k, n_rep)
        v = self._expand_kv(v, n_rep)

        # Build attention mask
        is_causal_flag = causal and past_kv is None
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            if tgt is not None:
                attn_mask = attn_mask.expand(-1, -1, T, -1)

        # SDPA rejects is_causal=True when attn_mask is also provided.
        # Fold the causal constraint directly into attn_mask instead.
        if is_causal_flag and attn_mask is not None:
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)
            is_causal_flag = False

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=is_causal_flag,
        )

        # (B, H, T, Dh) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(out.size(0), T, self.embed_dim)
        return self.out_proj(out), present_kv
