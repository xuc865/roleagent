"""Agent-In-World (AIW): failure history, string-similarity retrieval, and weighted training sampling."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizer

from role_agent.wia_utils import string_similarity

logger = logging.getLogger(__name__)

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
    # Path to dump failure records for visualization (None = no disk logging).
    dump_path: Optional[str] = None
    failed_history: List[Tuple[int, str]] = field(default_factory=list)
    weights: torch.Tensor = field(init=False)
    _training_step: int = field(default=0, init=False)

    def __post_init__(self):
        self.weights = torch.ones(self.num_tasks, dtype=torch.double)
        if self.dump_path is not None:
            os.makedirs(os.path.dirname(self.dump_path), exist_ok=True)

    def _trim_history(self) -> None:
        if len(self.failed_history) > self.max_history:
            self.failed_history = self.failed_history[-self.max_history :]

    def set_training_step(self, step: int) -> None:
        """Update the current training step counter (called by trainer each iteration)."""
        self._training_step = step

    def _extract_trajectory(
        self,
        steps: List[dict],
        tokenizer: PreTrainedTokenizer,
        max_action_chars: int = 500,
    ) -> List[Dict[str, str]]:
        """Extract a human-readable trajectory from a list of step dicts.

        Each step dict contains ``responses`` (token tensor for the agent
        action) and optionally ``prompts`` (token tensor for the observation
        fed to the model).  We decode both and return a compact list of
        ``{"action": ..., "observation_snippet": ...}`` dicts.
        """
        trajectory: List[Dict[str, str]] = []
        for step_dict in steps:
            action_text = ""
            obs_snippet = ""
            # Decode agent action from response tokens
            resp = step_dict.get("responses")
            if resp is not None:
                if hasattr(resp, "detach"):
                    resp = resp.detach().cpu()
                if resp.dim() > 1:
                    resp = resp[-1]
                action_text = tokenizer.decode(resp, skip_special_tokens=True)[:max_action_chars]

            # Decode a short snippet of the observation (last 400 chars of
            # the prompt) so we can see what the agent was looking at.
            prompt = step_dict.get("prompts")
            if prompt is not None:
                if hasattr(prompt, "detach"):
                    prompt = prompt.detach().cpu()
                if prompt.dim() > 1:
                    prompt = prompt[-1]
                full_obs = tokenizer.decode(prompt, skip_special_tokens=True)
                obs_snippet = full_obs[-400:]

            trajectory.append({
                "action": action_text,
                "observation_snippet": obs_snippet,
            })
        return trajectory

    def _dump_failure_record(
        self,
        task_idx: int,
        fingerprint: str,
        reward: float,
        trajectory: Optional[List[Dict[str, str]]] = None,
        episode_length: int = 0,
        data_source: str = "",
    ) -> None:
        """Append a single failure record (with full trajectory) to the .jsonl file."""
        if self.dump_path is None:
            return
        record: Dict[str, Any] = {
            "training_step": self._training_step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_idx": task_idx,
            "task_fingerprint": fingerprint[:200],
            "episode_reward": round(reward, 4),
            "weight_after": round(float(self.weights[task_idx]), 4),
            "history_size": len(self.failed_history),
            "episode_length": episode_length,
            "data_source": data_source,
        }
        if trajectory is not None:
            record["trajectory"] = trajectory
        try:
            with open(self.dump_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write failure record to %s", self.dump_path)

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

                # Extract full trajectory for failure analysis
                trajectory = self._extract_trajectory(steps, tokenizer)

                # Infer data_source from step info if available
                data_source = ""
                for step_dict in reversed(steps):
                    ds = step_dict.get("data_source", "")
                    if ds:
                        data_source = str(ds)
                        break

                # Dump this failure with full trajectory to disk
                self._dump_failure_record(
                    task_idx=idx,
                    fingerprint=fp,
                    reward=won,
                    trajectory=trajectory,
                    episode_length=len(steps),
                    data_source=data_source,
                )
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
