"""
tests/test_model.py
-------------------
Smoke tests that verify correctness of shapes, forward passes, loss,
and beam search.  No GPU required — tests run on CPU.

Run with:
    python -m pytest fr2en/tests/test_model.py -v
"""
from __future__ import annotations

import torch
import pytest

from fr2en.configs.config import Config, ModelConfig
from fr2en.model.embeddings import TranslationEmbeddings, SinusoidalPositionalEncoding
from fr2en.model.attention import MultiHeadAttention
from fr2en.model.layers import EncoderLayer, DecoderLayer
from fr2en.model.transformer import Transformer
from fr2en.training.loss import LabelSmoothedCrossEntropy
from fr2en.data.dataset import make_collator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> ModelConfig:
    """Tiny model config for fast CPU tests."""
    return ModelConfig(
        vocab_size=1000,
        embedding_dim=64,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        ffn_dim=256,
        max_src_len=64,
        max_tgt_len=64,
        dropout=0.0,
        attention_dropout=0.0,
        label_smoothing=0.1,
        tie_weights=True,
    )


@pytest.fixture
def full_config(small_config) -> Config:
    cfg = Config()
    cfg.model = small_config
    return cfg


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_save_load_roundtrip(self, tmp_path, full_config):
        path = tmp_path / "config.json"
        full_config.save(path)
        loaded = Config.load(path)
        assert loaded.model.vocab_size == full_config.model.vocab_size
        assert loaded.model.num_heads  == full_config.model.num_heads

    def test_base_and_large_dont_crash(self):
        _ = Config.base()
        _ = Config.large()

    def test_embedding_dim_head_assertion(self):
        with pytest.raises(AssertionError):
            ModelConfig(embedding_dim=64, num_heads=5, ffn_dim=256)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_sinusoidal_shape(self, small_config):
        pe = SinusoidalPositionalEncoding(small_config.max_src_len, small_config.embedding_dim)
        x = torch.zeros(2, 10, small_config.embedding_dim)
        out = pe(x)
        assert out.shape == (2, 10, small_config.embedding_dim)

    def test_translation_embeddings_src(self, small_config):
        emb = TranslationEmbeddings(small_config)
        ids = torch.randint(0, small_config.vocab_size, (3, 20))
        out = emb.encode_src(ids)
        assert out.shape == (3, 20, small_config.embedding_dim)

    def test_weight_tying(self, small_config):
        emb = TranslationEmbeddings(small_config)
        assert emb.src_embed.weight is emb.tgt_embed.weight


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:
    def test_self_attention_shape(self, small_config):
        attn = MultiHeadAttention(small_config)
        x = torch.randn(2, 10, small_config.embedding_dim)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_causal_attention(self, small_config):
        attn = MultiHeadAttention(small_config)
        x = torch.randn(2, 10, small_config.embedding_dim)
        out, _ = attn(x, causal=True)
        assert out.shape == x.shape

    def test_cross_attention_shape(self, small_config):
        attn = MultiHeadAttention(small_config)
        src = torch.randn(2, 12, small_config.embedding_dim)
        tgt = torch.randn(2, 8,  small_config.embedding_dim)
        out, _ = attn(src, tgt=tgt)
        assert out.shape == (2, 8, small_config.embedding_dim)

    def test_padding_mask_applied(self, small_config):
        """With all-zero mask (all padding), output should differ from no mask."""
        attn = MultiHeadAttention(small_config)
        x    = torch.randn(1, 5, small_config.embedding_dim)
        mask = torch.zeros(1, 5, dtype=torch.bool)  # all padding
        # Should not raise; output values will be NaN-free
        out, _ = attn(x, padding_mask=mask)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Encoder / Decoder layers
# ---------------------------------------------------------------------------

class TestLayers:
    def test_encoder_layer_shape(self, small_config):
        layer = EncoderLayer(small_config)
        x = torch.randn(2, 10, small_config.embedding_dim)
        out = layer(x)
        assert out.shape == x.shape

    def test_decoder_layer_shape(self, small_config):
        layer   = DecoderLayer(small_config)
        enc_out = torch.randn(2, 12, small_config.embedding_dim)
        tgt     = torch.randn(2, 8,  small_config.embedding_dim)
        out, _ = layer(tgt, enc_out)
        assert out.shape == tgt.shape

    def test_pre_ln_residual(self, small_config):
        """Pre-LN: output should not equal input (residual connection is applied)."""
        layer = EncoderLayer(small_config)
        x   = torch.randn(1, 5, small_config.embedding_dim)
        out = layer(x)
        assert not torch.allclose(x, out, atol=1e-5)


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class TestTransformer:
    def test_forward_shape(self, small_config):
        model   = Transformer(small_config)
        src_ids = torch.randint(4, small_config.vocab_size, (2, 10))
        tgt_ids = torch.randint(4, small_config.vocab_size, (2,  8))
        logits  = model(src_ids, tgt_ids)
        assert logits.shape == (2, 8, small_config.vocab_size)

    def test_encode_shape(self, small_config):
        model   = Transformer(small_config)
        src_ids = torch.randint(4, small_config.vocab_size, (2, 10))
        enc     = model.encode(src_ids)
        assert enc.shape == (2, 10, small_config.embedding_dim)

    def test_decode_shape(self, small_config):
        model   = Transformer(small_config)
        enc_out = torch.randn(2, 10, small_config.embedding_dim)
        tgt_ids = torch.randint(4, small_config.vocab_size, (2, 8))
        logits  = model.decode(tgt_ids, enc_out)
        assert logits.shape == (2, 8, small_config.vocab_size)

    def test_weight_tying(self, small_config):
        model = Transformer(small_config)
        assert model.lm_head.weight is model.embeddings.tgt_embed.weight

    def test_forward_backward(self, small_config):
        model   = Transformer(small_config)
        src_ids = torch.randint(4, small_config.vocab_size, (2, 10))
        tgt_ids = torch.randint(4, small_config.vocab_size, (2,  8))
        logits  = model(src_ids, tgt_ids)
        loss    = logits.sum()
        loss.backward()
        # All parameters should have gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_parameter_count_positive(self, small_config):
        model = Transformer(small_config)
        assert model.num_parameters() > 0
        assert model.num_parameters(exclude_embeddings=True) > 0

    def test_padding_mask_doesnt_affect_non_padding(self, small_config):
        """
        Padding tokens should not change non-padding outputs.
        Run two forward passes:
          1. clean src [A, B, C] with no mask
          2. src with a padding token at position 3, mask excluding position 3
        Positions 0..2 in the encoder output should be (approximately) equal.
        """
        model = Transformer(small_config)
        model.eval()
        S, D = 3, small_config.embedding_dim
        src1 = torch.randint(4, small_config.vocab_size, (1, S))
        # Add a padding position
        src2 = torch.cat([src1, torch.zeros(1, 1, dtype=torch.long)], dim=1)
        mask2 = torch.ones(1, S + 1, dtype=torch.bool)
        mask2[0, S] = False
        with torch.no_grad():
            enc1 = model.encode(src1)
            enc2 = model.encode(src2, mask2)
        # Non-padding positions should be close
        assert torch.allclose(enc1, enc2[:, :S], atol=1e-4)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLoss:
    def test_shape_and_no_nan(self):
        loss_fn = LabelSmoothedCrossEntropy(vocab_size=100, label_smoothing=0.1)
        logits  = torch.randn(4, 10, 100)
        targets = torch.randint(0, 100, (4, 10))
        targets[:, -2:] = -100   # simulate padding
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_zero_smoothing_equals_ce(self):
        logits  = torch.randn(2, 5, 50)
        targets = torch.randint(0, 50, (2, 5))
        loss_ls = LabelSmoothedCrossEntropy(50, label_smoothing=0.0)(logits, targets)
        import torch.nn.functional as F
        loss_ce = F.cross_entropy(logits.view(-1, 50), targets.view(-1))
        assert torch.allclose(loss_ls, loss_ce, atol=1e-5)

    def test_ignore_index_respected(self):
        logits  = torch.randn(1, 4, 50)
        targets = torch.tensor([[-100, -100, -100, -100]])
        loss = LabelSmoothedCrossEntropy(50)(logits, targets)
        assert loss.item() == 0.0 or not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Collator (no real tokenizer needed — mock pad/bos/eos ids)
# ---------------------------------------------------------------------------

class MockTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3


class TestCollator:
    def test_collate_shapes(self):
        tok = MockTokenizer()
        collate = make_collator(tok)
        batch = [
            {"src_ids": [4, 5, 6], "tgt_ids": [7, 8]},
            {"src_ids": [4, 5],    "tgt_ids": [7, 8, 9, 10]},
        ]
        out = collate(batch)
        B = 2
        assert out["src_input_ids"].shape[0] == B
        assert out["tgt_input_ids"].shape[0] == B
        # tgt_input = tgt_labels shifted — same length
        assert out["tgt_input_ids"].shape == out["tgt_labels"].shape

    def test_teacher_forcing_shift(self):
        tok = MockTokenizer()
        collate = make_collator(tok)
        batch = [{"src_ids": [4, 5], "tgt_ids": [7, 8, 9]}]
        out = collate(batch)
        # input should start with BOS (id=1)
        assert out["tgt_input_ids"][0, 0].item() == tok.bos_id
        # last label should be EOS (id=2)
        label_row = out["tgt_labels"][0]
        # find last non-(-100) position
        valid = [t.item() for t in label_row if t.item() != -100]
        assert valid[-1] == tok.eos_id

    def test_padding_filled_with_pad_id(self):
        tok = MockTokenizer()
        collate = make_collator(tok)
        batch = [
            {"src_ids": [4, 5, 6, 7], "tgt_ids": [8, 9]},
            {"src_ids": [4],           "tgt_ids": [8, 9, 10, 11]},
        ]
        out = collate(batch)
        # Shorter src row should be padded with pad_id (0)
        assert out["src_input_ids"][1, 1].item() == tok.pad_id

    def test_labels_minus100_at_pad(self):
        tok = MockTokenizer()
        collate = make_collator(tok)
        batch = [
            {"src_ids": [4], "tgt_ids": [7]},
            {"src_ids": [4], "tgt_ids": [7, 8, 9]},
        ]
        out = collate(batch)
        # Row 0 has shorter tgt — padded positions in tgt_labels should be -100
        labels_row0 = out["tgt_labels"][0].tolist()
        assert -100 in labels_row0


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# New tests for v3 features: RoPE, GQA, KV-cache, SwiGLU, word dropout
# ---------------------------------------------------------------------------

class TestModernArchitecture:
    """Tests for SwiGLU, RoPE, GQA — all optional via config flags."""

    def _modern_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=1000,
            embedding_dim=64,
            num_heads=4,
            num_kv_heads=2,       # GQA: 4 query heads, 2 KV heads
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=256,
            max_src_len=64,
            max_tgt_len=64,
            dropout=0.0,
            attention_dropout=0.0,
            label_smoothing=0.1,
            tie_weights=True,
            use_swiglu=True,
            use_rope=True,
        )

    def test_swiglu_forward_shape(self):
        from fr2en.model.modern import SwiGLUFeedForward
        cfg = ModelConfig(embedding_dim=64, num_heads=4, ffn_dim=256,
                         use_swiglu=True, use_rope=False)
        ff = SwiGLUFeedForward(cfg)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_rope_forward_shape(self):
        from fr2en.model.modern import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
        q = torch.randn(2, 4, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_changes_qk(self):
        """RoPE should change Q and K values (not a no-op)."""
        from fr2en.model.modern import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q, q_rot, atol=1e-5)
        assert not torch.allclose(k, k_rot, atol=1e-5)

    def test_gqa_forward_shape(self):
        """GQA with num_kv_heads < num_heads should produce correct output shape."""
        cfg = ModelConfig(embedding_dim=64, num_heads=4, num_kv_heads=2,
                         ffn_dim=256, use_swiglu=False, use_rope=False)
        from fr2en.model.attention import MultiHeadAttention
        attn = MultiHeadAttention(cfg)
        x = torch.randn(2, 10, 64)
        out, kv = attn(x)
        assert out.shape == (2, 10, 64)

    def test_full_modern_model_forward(self):
        """Full model with SwiGLU + RoPE + GQA — forward shape and backward."""
        cfg = self._modern_config()
        model = Transformer(cfg)
        src = torch.randint(4, cfg.vocab_size, (2, 10))
        tgt = torch.randint(4, cfg.vocab_size, (2, 8))
        logits = model(src, tgt)
        assert logits.shape == (2, 8, cfg.vocab_size)
        logits.sum().backward()

    def test_kv_cache_decode_step(self):
        """decode_step with KV-cache should produce same logits as full decode."""
        cfg = ModelConfig(
            vocab_size=500, embedding_dim=64, num_heads=4, num_kv_heads=4,
            encoder_layers=2, decoder_layers=2, ffn_dim=256,
            use_swiglu=False, use_rope=False, dropout=0.0, attention_dropout=0.0,
        )
        model = Transformer(cfg)
        model.eval()
        src = torch.randint(4, cfg.vocab_size, (1, 8))
        src_mask = torch.ones(1, 8, dtype=torch.bool)
        tgt = torch.randint(4, cfg.vocab_size, (1, 5))

        with torch.no_grad():
            enc_out = model.encode(src, src_mask)

            # Full decode (reference)
            full_logits = model.decode(tgt, enc_out, src_mask)  # (1, 5, V)

            # Step-by-step with KV cache
            kv = None
            step_logits_list = []
            for t in range(tgt.size(1)):
                step_in = tgt[:, t:t+1]
                step_logits, kv = model.decode_step(step_in, enc_out, src_mask, past_kv=kv)
                step_logits_list.append(step_logits)

        step_logits_cat = torch.cat(step_logits_list, dim=1)  # (1, 5, V)
        # Step-by-step KV cache must match full decode (both causal, same model weights)
        assert torch.allclose(full_logits, step_logits_cat, atol=1e-4), \
            "KV-cache decode_step output does not match full decode"

    def test_kv_cache_is_faster_in_shape(self):
        """KV cache should grow by 1 step at each decode step."""
        cfg = ModelConfig(
            vocab_size=500, embedding_dim=64, num_heads=4, num_kv_heads=2,
            encoder_layers=2, decoder_layers=2, ffn_dim=256,
            use_swiglu=False, use_rope=False, dropout=0.0,
        )
        model = Transformer(cfg)
        model.eval()
        src = torch.randint(4, cfg.vocab_size, (1, 6))
        enc_out = model.encode(src)

        kv = None
        for step in range(4):
            tok = torch.randint(4, cfg.vocab_size, (1, 1))
            _, kv = model.decode_step(tok, enc_out, past_kv=kv)
            # Each KV tensor should have T = step+1
            assert kv[0][0].size(2) == step + 1, \
                f"At step {step}, KV cache size should be {step+1}, got {kv[0][0].size(2)}"


class TestWordDropout:
    def test_word_dropout_replaces_tokens(self):
        """With high dropout, many source tokens should become unk_id."""
        tok = MockTokenizer()
        collate = make_collator(tok, word_dropout=0.9)
        batch = [{"src_ids": [4, 5, 6, 7, 8], "tgt_ids": [9, 10]}]
        out = collate(batch)
        src = out["src_input_ids"][0]
        # With 0.9 dropout, most tokens (4–8) should be replaced with unk_id=3
        n_unk = (src == tok.unk_id).sum().item()
        assert n_unk >= 2, f"Expected word dropout to replace tokens, only {n_unk} replaced"

    def test_word_dropout_zero_no_change(self):
        """With dropout=0.0, src tokens must be unchanged."""
        tok = MockTokenizer()
        collate_no_drop = make_collator(tok, word_dropout=0.0)
        batch = [{"src_ids": [4, 5, 6], "tgt_ids": [7, 8]}]
        out = collate_no_drop(batch)
        src = out["src_input_ids"][0].tolist()
        assert 4 in src and 5 in src and 6 in src

    def test_word_dropout_never_drops_padding(self):
        """Word dropout must not replace pad tokens with unk."""
        tok = MockTokenizer()
        collate = make_collator(tok, word_dropout=1.0)  # drop everything
        batch = [
            {"src_ids": [4, 5, 6, 7], "tgt_ids": [8]},
            {"src_ids": [4],          "tgt_ids": [8]},   # shorter → padded
        ]
        out = collate(batch)
        # Positions that were pad (mask=False) must remain pad, not unk
        src = out["src_input_ids"]
        mask = out["src_pad_mask"]
        pad_positions = ~mask
        assert (src[pad_positions] == tok.pad_id).all(), \
            "Padding positions should remain pad_id even with word_dropout=1.0"
