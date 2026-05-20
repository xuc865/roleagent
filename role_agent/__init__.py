# Role-Agent: WIA/AIW utilities, curriculum, and optional LLM prompt templates.
# See docs/role_agent_alignment.md.

from .paper_prompts import (
    PROMPT_COMPARE_PREDICTED_VS_ACTUAL,
    PROMPT_FAILURE_MODE_FROM_TRAJECTORY,
    PROMPT_RETRIEVE_SIMILAR_FAILURES,
)
from .wia_utils import WIA_PROMPT_SUFFIX, role_agent_flag, parse_predict_next, string_similarity, predicate_multiplier

__all__ = [
    "PROMPT_COMPARE_PREDICTED_VS_ACTUAL",
    "PROMPT_FAILURE_MODE_FROM_TRAJECTORY",
    "PROMPT_RETRIEVE_SIMILAR_FAILURES",
    "WIA_PROMPT_SUFFIX",
    "role_agent_flag",
    "parse_predict_next",
    "string_similarity",
    "predicate_multiplier",
]
