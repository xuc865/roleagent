"""World-In-Agent (WIA): ``<predict_next>`` parsing and string-based predicate rewards."""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Optional

_PRED_RE = re.compile(r"<predict_next>(.*?)</predict_next>", re.DOTALL | re.IGNORECASE)

# Appended to user-facing agent prompts when WIA is enabled (one-step prediction).
WIA_PROMPT_SUFFIX = """
Additionally, before </think> or before <action>, predict the **very next** environment
observation you expect to see after executing your chosen action. Put **only** that prediction inside
<predict_next> and </predict_next> (one short paragraph). Then continue with your usual reasoning and
<action> tags.
"""

def role_agent_cfg(config: Any) -> dict:
    try:
        from omegaconf import OmegaConf

        d = OmegaConf.select(config, "algorithm.role_agent")
        if d is None:
            return {}
        if isinstance(d, dict):
            return d
        return OmegaConf.to_container(d, resolve=True) or {}
    except Exception:
        return {}


def role_agent_flag(config: Any, key: str, default: bool = False) -> bool:
    cfg = role_agent_cfg(config)
    return bool(cfg.get(key, default))


def parse_predict_next(response: str) -> Optional[str]:
    m = _PRED_RE.search(response or "")
    if not m:
        return None
    return (m.group(1) or "").strip()


def string_similarity(
    a: Optional[str],
    b: Optional[str],
    *,
    max_chars: Optional[int] = None,
) -> float:
    """
    Text similarity in ``[0, 1]`` using the same ``difflib.SequenceMatcher`` ratio as GiGPO
    ``gigpo.core_gigpo.text_similarity_ratio`` / ``are_similar`` (no case folding).

    Implemented here without importing ``gigpo.core_gigpo`` so rollout workers avoid pulling
    ``torch`` for a pure-text helper.

    ``max_chars`` (when > 0) truncates both sides before comparison for cheaper rollouts on very
    long observations; GiGPO anchor grouping uses full strings (use ``0`` / omit to match that).
    """
    if not a or not b:
        return 0.0
    if max_chars is not None and max_chars > 0:
        a = a[:max_chars]
        b = b[:max_chars]
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def predicate_multiplier(sim: float) -> float:
    """Paper Eq.5 style: R_scaled = R_ori * sigmoid(R_pre); here R_pre is one-step similarity in [0,1]."""
    return 1.0 / (1.0 + math.exp(-float(sim)))


def next_obs_text_for_wia(next_obs: dict, batch_i: int) -> str:
    """Pick a stable string for the post-step observation (prefer raw anchor)."""
    anchor = next_obs.get("anchor")
    if anchor is not None:
        a = anchor[batch_i]
        if isinstance(a, str):
            return a
        if hasattr(a, "dtype") and hasattr(a, "shape"):
            return ""
        return str(a)
    text = next_obs.get("text")
    if text is not None:
        t = text[batch_i]
        if isinstance(t, str):
            return t
        return str(t)
    return ""
