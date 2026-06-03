"""
Dataset registry for the Token-Agent mixed benchmark.

Maps each data_source to a task_category and provides per-dataset
download / processing helpers.

Task categories (active in training):
  0  math_reasoning   AIME, GSM8K, MATH
  1  quick_qa         SQuAD
  2  direct_qa        SimpleQA, AA-Omniscience
  3  search_qa        NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
  4  action_env       ALFWorld, WebShop

Interface-only (not mixed into training):
  5  game_env         Sokoban, EZPoints
  6  multimodal       MMStar, SQA, MMVet, POPE, MMB, Math-V, AI2D
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

TASK_CATEGORIES = {
    0: "math_reasoning",
    1: "quick_qa",
    2: "direct_qa",
    3: "search_qa",
    4: "action_env",
    5: "game_env",
    6: "multimodal",
}

CATEGORY_NAME_TO_ID = {v: k for k, v in TASK_CATEGORIES.items()}


@dataclass
class DatasetMeta:
    data_source: str
    task_category: int
    hf_repo_id: Optional[str] = None
    split_map: Optional[Dict[str, str]] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Registry: data_source -> DatasetMeta
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, DatasetMeta] = {}


def _register(meta: DatasetMeta):
    DATASET_REGISTRY[meta.data_source] = meta


# ---- Math Reasoning (category 0) -----------------------------------------

_register(DatasetMeta(
    data_source="openai/gsm8k",
    task_category=0,
    hf_repo_id="openai/gsm8k",
    split_map={"train": "train", "test": "test"},
    description="Grade-school math word problems",
))

_register(DatasetMeta(
    data_source="lighteval/MATH",
    task_category=0,
    # HF 上该数据集的实际 repo id 可能为 DigitalLearningGmbH/MATH-lighteval；
    # 避免 preprocess 阶段 DatasetNotFoundError。
    hf_repo_id="DigitalLearningGmbH/MATH-lighteval",
    split_map={"train": "train", "test": "test"},
    description="Competition-level math problems",
))

_register(DatasetMeta(
    data_source="math_dapo",
    task_category=0,
    hf_repo_id=None,
    description="AIME-style math (DAPO format)",
))

for aime_year in ["aime_2024", "aime_2025"]:
    _register(DatasetMeta(
        data_source=aime_year,
        task_category=0,
        description=f"AIME {aime_year[-4:]} problems",
    ))

# ---- Quick QA (category 1) -----------------------------------------------

_register(DatasetMeta(
    data_source="squad",
    task_category=1,
    hf_repo_id="rajpurkar/squad",
    split_map={"train": "train", "test": "validation"},
    description="Stanford Question Answering Dataset (extractive QA)",
))

# ---- Direct QA (category 2) ----------------------------------------------

_register(DatasetMeta(
    data_source="simpleqa",
    task_category=2,
    hf_repo_id="/mnt/workspace/wxc/Agent/otherdata/simpleqa",
    split_map={"test": "test"},
    description="OpenAI SimpleQA – factual direct-answer questions",
)) 


# ---- Search QA (category 3) ----------------------------------------------
  
_register(DatasetMeta(
    data_source="search_qa",
    task_category=3,
    hf_repo_id="/mnt/workspace/wxc/Agent/otherdata/search_qa",
    split_map={"train": "train", "test": "test"},
    description=f"Search-R1 style QA",
))

# ---- Action Env (category 4) ---------------------------------------------

_register(DatasetMeta(
    data_source="alfworld",
    task_category=4,
    description="ALFWorld embodied household tasks",
))

# "text" is the data_source produced by examples.data_preprocess.prepare
# when running in text-only mode (e.g. for alfworld standalone training).
_register(DatasetMeta(
    data_source="text",
    task_category=4,
    description="Generic text-mode data (alfworld standalone)",
))

_register(DatasetMeta(
    data_source="webshop",
    task_category=4,
    description="WebShop e-commerce navigation",
))

# ---- Game Env (category 5, interface only) --------------------------------

_register(DatasetMeta(
    data_source="sokoban",
    task_category=5,
    description="Sokoban puzzle game",
))

_register(DatasetMeta(
    data_source="ezpoints",
    task_category=5,
    description="EZPoints card math game",
))

# ---- Multimodal (category 6, interface only) ------------------------------

for mm_name in ["mmstar", "sqa", "mmvet", "pope", "mmb", "math_v", "ai2d"]:
    _register(DatasetMeta(
        data_source=mm_name,
        task_category=6,
        description=f"Multimodal benchmark: {mm_name}",
    ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_task_category(data_source: str) -> int:
    """Return the task_category int for a given data_source string."""
    if data_source in DATASET_REGISTRY:
        return DATASET_REGISTRY[data_source].task_category
    for prefix in ["searchR1_", "aime"]:
        if data_source.startswith(prefix):
            return 3 if prefix == "searchR1_" else 0
    raise KeyError(f"Unknown data_source: {data_source}")


def get_active_datasets() -> List[DatasetMeta]:
    """Return datasets with task_category 0-4 (mixed into training)."""
    return [m for m in DATASET_REGISTRY.values() if m.task_category <= 4]


def get_interface_only_datasets() -> List[DatasetMeta]:
    """Return datasets with task_category >= 5 (interface only)."""
    return [m for m in DATASET_REGISTRY.values() if m.task_category >= 5]
