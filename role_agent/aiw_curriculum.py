"""Agent-In-World (AIW): failure history, string-similarity retrieval, and weighted training sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizer

from role_agent.wia_utils import string_similarity


def _decode_prompt(step0: dict, tokenizer: PreTrainedTokenizer, max_chars: int = 800) -> str:
    p = step0.get("prompts")
    if p is None:
        return ""
    if hasattr(p, "detach"):
        p = p.detach().cpu()
    ids = p
    if ids.dim() > 1:
        ids = ids[-1]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text[:max_chars]


@dataclass
class AIWCurriculum:
    """Keeps failure fingerprints and a mutable weight vector aligned with dataset indices."""

    num_tasks: int
    top_k: int = 3
    boost: float = 0.35
    self_boost: float = 0.15
    max_history: int = 512
    # GiGPO-style gate: only count cross-task boosts when sim >= threshold (0 disables).
    similarity_thresh: float = 0.0
    # Truncate fingerprints before difflib (0 = full string, same regime as GiGPO anchors).
    text_match_max_chars: int = 0
    failed_history: List[Tuple[int, str]] = field(default_factory=list)
    weights: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.weights = torch.ones(self.num_tasks, dtype=torch.double)

    def _trim_history(self) -> None:
        if len(self.failed_history) > self.max_history:
            self.failed_history = self.failed_history[-self.max_history :]

    def record_rollout_end(
        self,
        total_batch_list: List[List[dict]],
        success: dict[str, np.ndarray],
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        sr = success.get("success_rate")
        if sr is None:
            return
        sr = np.asarray(sr).reshape(-1)
        for bs, steps in enumerate(total_batch_list):
            if not steps:
                continue
            if bs >= len(sr):
                break
            d0 = steps[0]
            try:
                idx = int(d0["index"])
            except (TypeError, ValueError, KeyError):
                continue
            if idx < 0 or idx >= self.num_tasks:
                continue
            fp = _decode_prompt(d0, tokenizer)
            won = float(sr[bs])
            if won < 1.0:
                self.failed_history.append((idx, fp))
                self._trim_history()
                self.weights[idx] += self.self_boost
                if len(self.failed_history) > 1:
                    scored: List[Tuple[int, float]] = []
                    mx = self.text_match_max_chars
                    for _, (hidx, hfp) in enumerate(self.failed_history[:-1]):
                        if hidx == idx:
                            continue
                        sim = string_similarity(fp, hfp, max_chars=mx if mx > 0 else None)
                        if self.similarity_thresh > 0.0 and sim < self.similarity_thresh:
                            continue
                        scored.append((hidx, sim))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    seen = set()
                    for hidx, sim in scored:
                        if sim <= 0.0 or hidx in seen:
                            continue
                        seen.add(hidx)
                        self.weights[hidx] += self.boost * sim
                        if len(seen) >= self.top_k:
                            break
        self.weights.clamp_(min=1e-3)


class MutableWeightedSampler(Sampler[int]):
    """Yields `num_samples` indices per epoch; probabilities follow `weights_buf` (updated in-place)."""

    def __init__(self, num_samples: int, weights_buf: torch.Tensor, generator: Optional[torch.Generator] = None):
        self.num_samples = num_samples
        self.weights_buf = weights_buf
        self.generator = generator

    def __iter__(self):
        w = self.weights_buf
        if w.sum() <= 0 or math.isnan(float(w.sum())):
            w = torch.ones_like(w)
        idx = torch.multinomial(w, self.num_samples, replacement=True, generator=self.generator)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples
