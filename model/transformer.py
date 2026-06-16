"""
model/transformer.py  (v3)
--------------------------
Changes from v2:
  - encode() / decode() handle new (output, present_kv) signature from layers
  - New decode_step() for single-token KV-cache inference
  - embeddings skips PE addition when use_rope=True (RoPE is applied in attention)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from fr2en.configs.config import ModelConfig
from fr2en.model.embeddings import TranslationEmbeddings
from fr2en.model.layers import DecoderLayer, EncoderLayer

logger = logging.getLogger(__name__)

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]  # one (K, V) per decoder layer


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = TranslationEmbeddings(config)

        use_ckpt = config.gradient_checkpointing
        self.encoder = nn.ModuleList(
            [EncoderLayer(config, use_checkpoint=use_ckpt) for _ in range(config.encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.embedding_dim)

        self.decoder = nn.ModuleList(
            [DecoderLayer(config, use_checkpoint=use_ckpt) for _ in range(config.decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(config.embedding_dim)

        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embeddings.tgt_embed.weight

        self.apply(self._init_weights)
        with torch.no_grad():
            self.embeddings.src_embed.weight[config.pad_token_id].zero_()
            if not config.tie_weights:
                self.embeddings.tgt_embed.weight[config.pad_token_id].zero_()

        logger.info(
            "Transformer | %.2fM params | %.2fM non-embed | swiglu=%s rope=%s kv_heads=%d/%d",
            self.num_parameters() / 1e6,
            self.num_parameters(exclude_embeddings=True) / 1e6,
            config.use_swiglu, config.use_rope,
            config.num_kv_heads, config.num_heads,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        if not exclude_embeddings:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed_ids = {id(p) for p in self.embeddings.parameters()}
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad and id(p) not in embed_ids
        )

    # ------------------------------------------------------------------
    # Training forward (full sequence)
    # ------------------------------------------------------------------

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src_input_ids, src_pad_mask)
        return self.decode(tgt_input_ids, enc_out, src_pad_mask, tgt_pad_mask)

    # ------------------------------------------------------------------
    # Encoder (called once per source sentence)
    # ------------------------------------------------------------------

    def encode(
        self,
        src_input_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embeddings.encode_src(src_input_ids)
        for layer in self.encoder:
            x = layer(x, src_mask=src_pad_mask)
        return self.encoder_norm(x)

    # ------------------------------------------------------------------
    # Decoder — full sequence (training / teacher forcing)
    # ------------------------------------------------------------------

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embeddings.encode_tgt(tgt_input_ids)
        for layer in self.decoder:
            x, _ = layer(x, enc_out, src_mask=src_pad_mask, tgt_mask=tgt_pad_mask)
        return self.lm_head(self.decoder_norm(x))

    # ------------------------------------------------------------------
    # Decoder — single step with KV-cache (inference)
    # ------------------------------------------------------------------

    def decode_step(
        self,
        tgt_ids_step: torch.Tensor,          # (B, 1) — latest token only
        enc_out: torch.Tensor,               # (B, S, D) — encoder output, unchanged
        src_pad_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,   # list of (K, V) per layer
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Single decode step for KV-cache inference.

        Returns
        -------
        logits     : (B, 1, V)
        present_kv : updated KV cache (list of (K, V) per layer)
        """
        # step_offset = number of tokens already in the KV cache
        # so the new token gets PE at position step_offset, not position 0
        step_offset = past_kv[0][0].size(2) if past_kv is not None else 0
        x = self.embeddings.encode_tgt(tgt_ids_step, step_offset=step_offset)
        present_kv: KVCache = []
        for i, layer in enumerate(self.decoder):
            layer_past = past_kv[i] if past_kv is not None else None
            x, new_kv = layer(x, enc_out, src_mask=src_pad_mask, past_self_kv=layer_past)
            present_kv.append(new_kv)
        return self.lm_head(self.decoder_norm(x)), present_kv
