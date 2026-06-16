"""
inference/beam_search.py  (v3 — true batched beam search)
----------------------------------------------------------
Key change from v2:
  ALL source sentences are encoded in one forward pass and the decoder runs
  (B × beam_size) candidates simultaneously at every step.

  v2 (sequential): O(B) encoder calls, GPU at ~5% utilisation during eval
  v3 (batched):    O(1) encoder call, full GPU utilisation for all B*beam paths

Algorithm
---------
  Outer loop : decode steps (up to max_len tokens)
  Inner state : (B*beam) sequences sharing encoder outputs expanded B*beam times
  KV cache   : (B*beam, num_kv_heads, T, head_dim) — one cache per decoder layer
  Completion : when a beam hits EOS its hypothesis is stored in completed[b];
               the vacated slot is replaced with a dead-beam placeholder
               (score=-inf) so tensor shapes stay constant

The single-sentence beam_search() is kept for standalone / interactive use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from fr2en.configs.config import InferenceConfig


@dataclass
class BeamHypothesis:
    token_ids: List[int] = field(default_factory=list)
    score: float = 0.0

    def length_normed_score(self, alpha: float) -> float:
        if not self.token_ids:
            return self.score
        lp = ((5 + len(self.token_ids)) / 6) ** alpha
        return self.score / lp


# ---------------------------------------------------------------------------
# Batched beam search  (main entry point)
# ---------------------------------------------------------------------------

def batch_beam_search(
    model,
    src_ids_list: List[torch.Tensor],
    tokenizer,
    config: InferenceConfig,
    device: torch.device,
) -> List[List[int]]:
    """
    Encode all source sentences in ONE forward pass, then decode
    B × beam_size candidates in parallel at every step.

    Parameters
    ----------
    src_ids_list : list of (S_i,) or (1, S_i) source-token tensors (variable length).
    tokenizer    : SharedTokenizer — used for special token ids.
    config       : InferenceConfig (beam_size, length_penalty, …).
    device       : target device.

    Returns
    -------
    List of token-id lists, one per input sentence (no BOS / EOS).
    """
    B = len(src_ids_list)
    if B == 0:
        return []

    beam  = config.beam_size
    alpha = config.length_penalty
    bos   = tokenizer.bos_id
    eos   = tokenizer.eos_id
    pad   = tokenizer.pad_id

    # ------------------------------------------------------------------
    # 1. Pad all sources to a common length and run the encoder once
    # ------------------------------------------------------------------
    src_flat = [s.to(device).view(-1) for s in src_ids_list]
    max_src  = max(s.size(0) for s in src_flat)

    src_padded = torch.full((B, max_src), pad, dtype=torch.long, device=device)
    for i, s in enumerate(src_flat):
        src_padded[i, : s.size(0)] = s
    src_mask = src_padded.ne(pad)                         # (B, max_src)

    with torch.no_grad():
        enc_out = model.encode(src_padded, src_mask)      # (B, max_src, D)

    S, D = enc_out.size(1), enc_out.size(2)

    # Expand for (B*beam) parallel decode paths
    enc_beam = enc_out.unsqueeze(1).expand(B, beam, S, D).reshape(B * beam, S, D)
    msk_beam = src_mask.unsqueeze(1).expand(B, beam, S).reshape(B * beam, S)

    # ------------------------------------------------------------------
    # 2. Initialize all (B*beam) beams
    # ------------------------------------------------------------------
    tgt_ids = torch.full((B * beam, 1), bos, dtype=torch.long, device=device)

    # First beam per sentence: score=0; rest: score=-inf (not yet active)
    beam_scores = torch.full((B, beam), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(B * beam)              # (B*beam,)

    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    completed: List[List[BeamHypothesis]] = [[] for _ in range(B)]
    done      = [False] * B

    max_len = int(config.max_len_a * max_src + config.max_len_b)

    # ------------------------------------------------------------------
    # 3. Decode step by step
    # ------------------------------------------------------------------
    with torch.no_grad():
        for _ in range(max_len):
            if all(done):
                break

            latest = tgt_ids[:, -1:]                           # (B*beam, 1)
            logits, kv_cache = model.decode_step(
                latest, enc_beam, msk_beam, past_kv=kv_cache
            )
            logits_step = logits[:, 0, :]                      # (B*beam, V)

            if config.no_repeat_ngram_size > 0:
                logits_step = _suppress_ngrams(
                    logits_step, tgt_ids, config.no_repeat_ngram_size
                )

            log_probs = F.log_softmax(logits_step, dim=-1)    # (B*beam, V)
            V = log_probs.size(-1)

            # Candidate scores reshaped to (B, beam*V) for per-sentence topk
            cand = (beam_scores.unsqueeze(1) + log_probs).view(B, beam * V)
            top_scores, top_flat = cand.topk(2 * beam, dim=1) # (B, 2*beam)
            top_beams  = top_flat // V                         # parent beam index
            top_tokens = top_flat %  V                         # token index

            next_tgt    : List[torch.Tensor] = []
            next_scores : List[float]        = []
            next_kv_sel : List[int]          = []

            for b in range(B):
                if done[b]:
                    # Keep placeholder rows to maintain (B*beam) shape.
                    # Append dummy token (+1 length) and -inf score.
                    g0 = b * beam
                    for k in range(beam):
                        g = b * beam + k
                        dummy = torch.cat(
                            [tgt_ids[g], torch.tensor([pad], device=device)]
                        )
                        next_tgt.append(dummy)
                        next_scores.append(float("-inf"))
                        next_kv_sel.append(g0)       # all point to first beam
                    continue

                filled = 0
                for rank_i in range(2 * beam):
                    if filled >= beam:
                        break
                    b_beam = top_beams [b, rank_i].item()
                    tok    = top_tokens[b, rank_i].item()
                    score  = top_scores[b, rank_i].item()
                    g      = b * beam + b_beam

                    if tok == eos:
                        # Record completed hypothesis (strip BOS at position 0)
                        seq = tgt_ids[g, 1:].tolist()
                        completed[b].append(BeamHypothesis(token_ids=seq, score=score))
                    else:
                        new_row = torch.cat(
                            [tgt_ids[g], torch.tensor([tok], device=device)]
                        )
                        next_tgt.append(new_row)
                        next_scores.append(score)
                        next_kv_sel.append(g)
                        filled += 1

                # If some beams ended in EOS, pad with dead-beam placeholders
                # so this sentence always contributes exactly `beam` rows.
                g0 = b * beam
                while filled < beam:
                    dummy = torch.cat(
                        [tgt_ids[g0], torch.tensor([pad], device=device)]
                    )
                    next_tgt.append(dummy)
                    next_scores.append(float("-inf"))
                    next_kv_sel.append(g0)
                    filled += 1

                if config.early_stopping and len(completed[b]) >= beam:
                    done[b] = True

            # Stack next state — all rows are now T+1 length
            tgt_ids     = torch.stack(next_tgt)               # (B*beam, T+1)
            beam_scores = torch.tensor(next_scores, device=device)

            # Reorder KV cache to match surviving / placeholder beam order
            if kv_cache is not None:
                sel_t    = torch.tensor(next_kv_sel, dtype=torch.long, device=device)
                kv_cache = [(k[sel_t], v[sel_t]) for k, v in kv_cache]

    # ------------------------------------------------------------------
    # 4. Pick best hypothesis per sentence
    # ------------------------------------------------------------------
    results: List[List[int]] = []
    for b in range(B):
        if completed[b]:
            best = max(completed[b], key=lambda h: h.length_normed_score(alpha))
            results.append(best.token_ids)
        else:
            # No hypothesis completed — return best active beam (strip BOS)
            slab  = beam_scores[b * beam : (b + 1) * beam]
            best_k = int(slab.argmax().item())
            results.append(tgt_ids[b * beam + best_k, 1:].tolist())

    return results


# ---------------------------------------------------------------------------
# Single-sentence beam search (kept for interactive / scripted use)
# ---------------------------------------------------------------------------

def beam_search(
    model,
    src_ids: torch.Tensor,                  # (1, S)
    src_pad_mask: Optional[torch.Tensor],
    tokenizer,
    config: InferenceConfig,
    device: torch.device,
) -> List[int]:
    """
    Beam search for a single source sentence.  Returns token ids (no BOS/EOS).
    Delegates to batch_beam_search so both code paths stay in sync.
    """
    return batch_beam_search(
        model, [src_ids.squeeze(0)], tokenizer, config, device
    )[0]


# ---------------------------------------------------------------------------
# No-repeat ngram suppression
# ---------------------------------------------------------------------------

def _suppress_ngrams(
    logits: torch.Tensor,   # (B*beam, V)
    tgt_ids: torch.Tensor,  # (B*beam, T) — includes BOS at position 0
    ngram_size: int,
) -> torch.Tensor:
    logits = logits.clone()
    for b in range(tgt_ids.size(0)):
        gen = tgt_ids[b, 1:].tolist()   # skip BOS
        if len(gen) < ngram_size - 1:
            continue
        prefix = tuple(gen[-(ngram_size - 1):])
        for i in range(len(gen) - ngram_size + 1):
            if tuple(gen[i : i + ngram_size - 1]) == prefix:
                logits[b, gen[i + ngram_size - 1]] = float("-inf")
    return logits
