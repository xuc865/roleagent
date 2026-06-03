"""
Preprocess all active datasets into a unified mixed-benchmark parquet.

Usage:
    python -m token_agent.data.preprocess_mixed_benchmark \
        --local_dir ~/data/token_agent_mixed \
        --active_categories 0,1,2,3

Output columns per row:
    data_source      str              e.g. "openai/gsm8k", "squad", "searchR1_nq"
    task_category    int              0-4
    prompt           list[dict]       chat-format messages (system + user)
    env_kwargs       dict             env-specific kwargs (ground_truth, question, …)
    reward_model     dict | None      ground-truth payload for reward scoring
    extra_info       dict             index, split, tools_kwargs, …
"""

import argparse
import json
import logging
import os
import glob
from typing import Callable, Dict, List, Optional

import pandas as pd

from token_agent.data.dataset_registry import (
    DATASET_REGISTRY,
    DatasetMeta,
    get_active_datasets,
)
from token_agent.prompts.unified_system_prompt import UNIFIED_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOCAL_DATA_DIR: Optional[str] = None

# ---------------------------------------------------------------------------
# Per-dataset output format instructions (format only, no reasoning hints)
# ---------------------------------------------------------------------------

_FORMAT_INSTRUCTIONS: Dict[str, str] = {
    # Math reasoning
    "openai/gsm8k": "IMPORTANT: Do NOT use <answer> tags. You MUST end your response with '#### <number>' (the number only, no units), for example: #### 42",
    "lighteval/MATH": "IMPORTANT: Do NOT use <answer> tags. You MUST present your final answer inside \\boxed{}, for example: \\boxed{42}",
    "math_dapo": "IMPORTANT: Do NOT use <answer> tags. You MUST present your final answer inside \\boxed{}, for example: \\boxed{42}",
    "aime_2024": "IMPORTANT: Do NOT use <answer> tags. You MUST present your final answer inside \\boxed{}, for example: \\boxed{42}",
    "aime_2025": "IMPORTANT: Do NOT use <answer> tags. You MUST present your final answer inside \\boxed{}, for example: \\boxed{42}",
    # Extractive QA
    "squad": "Wrap your answer in <answer> tags, for example: <answer>Paris</answer>",
    # Direct / search QA
    "simpleqa": "Wrap your answer in <answer> tags, for example: <answer>Paris</answer>",
    "search_qa": "Wrap your answer in <answer> tags, for example: <answer>Paris</answer>",
    # Action environments
    "alfworld": "Wrap each action in <action> tags, for example: <action>go to desk 1</action>",
    "webshop": "Wrap each action in <action> tags, for example: <action>click[Buy Now]</action>",
}

# ---------------------------------------------------------------------------
# Per-dataset processors: each returns a list[dict] of unified rows
# ---------------------------------------------------------------------------

def _build_prompt(
    question: str,
    system: str = UNIFIED_SYSTEM_PROMPT,
    data_source: Optional[str] = None,
) -> List[dict]:
    """Build chat-format prompt list. Injects UNIFIED_SYSTEM_PROMPT by default.

    If ``data_source`` is provided and has a registered format instruction,
    appends a single-line output format requirement to the user message.
    The instruction describes only the required output format – no reasoning hints.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    format_hint = _FORMAT_INSTRUCTIONS.get(data_source, "") if data_source else ""
    user_content = f"{question}\n\n{format_hint}" if format_hint else question
    msgs.append({"role": "user", "content": user_content})
    return msgs

def _build_question_with_hint(question: str, data_source: Optional[str] = None) -> str:
    """Return the question string with the format hint appended (if any).

    This is used to populate ``env_kwargs["question"]`` so that
    ``SingleTurnEnvWrapper.reset()`` returns an observation that already
    contains the output-format instruction.  Without this, the format hint
    stored in the ``prompt`` field would be silently dropped because the
    rollout loop reconstructs the prompt from the env observation rather
    than from the parquet ``prompt`` column.
    """
    format_hint = _FORMAT_INSTRUCTIONS.get(data_source, "") if data_source else ""
    return f"{question}\n\n{format_hint}" if format_hint else question

def _ensure_system_prompt(row: dict) -> None:
    """Ensure a row's ``prompt`` list starts with the unified system message.

    Modifies in-place. Handles rows from pre-processed parquet files that
    may have been built without UNIFIED_SYSTEM_PROMPT.
    """
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) == 0:
        return
    first = prompt[0]
    if isinstance(first, dict) and first.get("role") == "system":
        return
    row["prompt"] = [{"role": "system", "content": UNIFIED_SYSTEM_PROMPT}] + prompt


def process_gsm8k(meta: DatasetMeta, split: str) -> List[dict]:
    # Priority order:
    # 1) If local split-specific files exist, use them.
    # 2) Otherwise, use HF's official train/test split.
    # 3) Only if HF fails, fall back to ratio split (90/10) on local raw json.
    if LOCAL_DATA_DIR:
        rows_local_split = _process_gsm8k_local_split_specific(meta=meta, split=split)
        if rows_local_split:
            return rows_local_split

    from datasets import load_dataset
    try:
        ds = load_dataset(meta.hf_repo_id, "main", split=meta.split_map[split])
        rows = []
        for i, ex in enumerate(ds):
            answer = ex["answer"].split("####")[-1].strip()
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(ex["question"], data_source=meta.data_source),
                "env_kwargs": {
                    "ground_truth": answer,
                    "question": _build_question_with_hint(ex["question"], data_source=meta.data_source),
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                },
                "reward_model": {"ground_truth": answer},
                "extra_info": {"index": i, "split": split},
            })
        return rows
    except Exception as e:
        logger.warning("Could not load %s from HF: %s.", meta.data_source, e)

    # HF failed: only then use local 90/10 split fallback.
    if LOCAL_DATA_DIR:
        rows_local_ratio = _process_gsm8k_local(meta=meta, split=split)
        if rows_local_ratio:
            logger.warning(
                "Falling back to local ratio split for %s (split=%s).", meta.data_source, split
            )
        return rows_local_ratio

    return []


def process_math(meta: DatasetMeta, split: str) -> List[dict]:
    # Priority order:
    # 1) If local split-specific files exist, use them.
    # 2) Otherwise, use HF's official split when hf_repo_id is available.
    # 3) Only if HF fails (or hf_repo_id is None), fall back to local ratio split.
    if LOCAL_DATA_DIR:
        rows_local_split = _process_math_local_split_specific(meta=meta, split=split)
        if rows_local_split:
            return rows_local_split

    if meta.hf_repo_id is not None:
        from datasets import load_dataset
        try:
            ds = load_dataset(meta.hf_repo_id, split=meta.split_map[split])
            rows = []
            for i, ex in enumerate(ds):
                rows.append({
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                    "prompt": _build_prompt(ex["problem"], data_source=meta.data_source),
                    "env_kwargs": {
                        "ground_truth": ex["solution"],
                        "question": _build_question_with_hint(ex["problem"], data_source=meta.data_source),
                        "data_source": meta.data_source,
                        "task_category": meta.task_category,
                    },
                    "reward_model": {"ground_truth": ex["solution"]},
                    "extra_info": {"index": i, "split": split},
                })
            return rows
        except Exception as e:
            logger.warning(
                "Could not load %s (hf_repo_id=%s): %s. Falling back to local ratio split.",
                meta.data_source,
                meta.hf_repo_id,
                e,
            )

    if LOCAL_DATA_DIR:
        return _process_math_local(meta=meta, split=split)

    return []


def _find_first_existing_file(root: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return None


def _jsonl_iter(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                yield json.loads(line)


def _local_dir_key(data_source: str) -> str:
    """
    Map a dataset `data_source` (as in DATASET_REGISTRY) to a preferred local
    sub-directory name under TOKEN_AGENT_DATA_DIR.

    Goal: avoid mixing raw datasets in one folder.
    """
    # gsm8k
    if data_source in ("openai/gsm8k", "gsm8k"):
        return "gsm8k"

    # math
    if data_source in ("lighteval/MATH", "math"):
        return "math"
    if data_source in ("math_dapo", "math_dapo_alt"):
        # you can choose to name this subdir as you like; keep a sensible default.
        return "math_dapo"

    # AIME
    if data_source.startswith("aime_"):
        # aime_2024 -> aime24 ; aime_2025 -> aime25
        year = data_source.split("_")[-1]
        suffix = year[-2:]  # "24" / "25"
        return f"aime{suffix}"
    if data_source in ("aime24", "aime25"):
        return data_source

    # SQuAD
    if data_source == "squad":
        return "squad"

    # Keep other datasets in their own directory using a normalized name.
    return data_source.replace("/", "_")


def _local_root_for_data_source(data_source: str) -> str:
    """
    Prefer TOKEN_AGENT_DATA_DIR/<local_subdir> if it exists, otherwise fall
    back to TOKEN_AGENT_DATA_DIR.
    """
    assert LOCAL_DATA_DIR is not None
    key = _local_dir_key(data_source)
    subdir = os.path.join(LOCAL_DATA_DIR, key)
    if os.path.isdir(subdir):
        return subdir
    return LOCAL_DATA_DIR


def _process_gsm8k_local(meta: DatasetMeta, split: str) -> List[dict]:
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)
    path = _find_first_existing_file(root, ["gsm8k.json", "gsm8k.jsonl"])
    if not path:
        return []

    # gsm8k.json is a JSON array of {"text": ..., "label": ...}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []

    n = len(data)
    if n == 0:
        return []
    cut = int(0.9 * n)

    rows = []
    for i, ex in enumerate(data):
        is_train = i < cut
        if split == "train" and not is_train:
            continue
        if split == "test" and is_train:
            continue

        text = ex.get("text") or ex.get("question") or ex.get("problem") or ""
        label = ex.get("label") if "label" in ex else ex.get("answer")
        if text == "" or label is None:
            continue
        answer = str(label).strip()
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(text, data_source=meta.data_source),
            "env_kwargs": {
                "ground_truth": answer,
                "question": text,
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def _process_gsm8k_local_split_specific(meta: DatasetMeta, split: str) -> List[dict]:
    """Try to load gsm8k from local *split-specific* files.

    This is used to prioritize official train/test splits when they exist locally.
    If only an unsplit raw file is available (e.g. gsm8k.json), this returns [] so
    that we can fall back to HF (or ratio split later).
    """
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)

    # Common local layouts produced by other scripts: train.parquet/test.parquet or
    # train.jsonl/test.jsonl under the dataset folder.
    parquet_candidates = [
        f"{split}.parquet",
        f"gsm8k_{split}.parquet",
        f"gsm8k-{split}.parquet",
        f"*{split}*.parquet",
    ]
    jsonl_candidates = [
        f"{split}.jsonl",
        f"gsm8k_{split}.jsonl",
        f"gsm8k-{split}.jsonl",
        f"*{split}*.jsonl",
    ]
    json_candidates = [
        f"{split}.json",
        f"gsm8k_{split}.json",
        f"gsm8k-{split}.json",
        f"*{split}*.json",
    ]

    selected_path: Optional[str] = None
    selected_kind: str = ""

    for pat in parquet_candidates:
        matches = glob.glob(os.path.join(root, pat))
        if matches:
            selected_path = matches[0]
            selected_kind = "parquet"
            break
    if selected_path is None:
        for pat in jsonl_candidates:
            matches = glob.glob(os.path.join(root, pat))
            if matches:
                selected_path = matches[0]
                selected_kind = "jsonl"
                break
    if selected_path is None:
        for pat in json_candidates:
            matches = glob.glob(os.path.join(root, pat))
            if matches:
                selected_path = matches[0]
                selected_kind = "json"
                break

    if not selected_path:
        return []

    rows: List[dict] = []
    if selected_kind == "parquet":
        df = pd.read_parquet(selected_path)
        # Heuristics for column naming.
        q_col = next((c for c in ["question", "problem", "text"] if c in df.columns), None)
        a_col = next((c for c in ["answer", "solution", "label"] if c in df.columns), None)
        if q_col is None or a_col is None:
            return []
        for i, ex in enumerate(df.to_dict(orient="records")):
            question = ex.get(q_col) or ""
            label = ex.get(a_col)
            if question == "" or label is None:
                continue
            answer = str(label).strip()
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question, data_source=meta.data_source),
                "env_kwargs": {
                    "ground_truth": answer,
                    "question": _build_question_with_hint(question, data_source=meta.data_source),
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                },
                "reward_model": {"ground_truth": answer},
                "extra_info": {"index": i, "split": split},
            })
        return rows

    if selected_kind == "jsonl":
        for i, ex in enumerate(_jsonl_iter(selected_path)):
            question = ex.get("question") or ex.get("problem") or ex.get("text") or ""
            label = (
                ex.get("answer")
                if "answer" in ex
                else ex.get("solution")
                if "solution" in ex
                else ex.get("label")
            )
            if question == "" or label is None:
                continue
            answer = str(label).strip()
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question, data_source=meta.data_source),
                "env_kwargs": {
                    "ground_truth": answer,
                    "question": _build_question_with_hint(question, data_source=meta.data_source),
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                },
                "reward_model": {"ground_truth": answer},
                "extra_info": {"index": i, "split": split},
            })
        return rows

    # selected_kind == "json"
    with open(selected_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    for i, ex in enumerate(data):
        question = ex.get("question") or ex.get("problem") or ex.get("text") or ""
        label = (
            ex.get("answer")
            if "answer" in ex
            else ex.get("solution")
            if "solution" in ex
            else ex.get("label")
        )
        if question == "" or label is None:
            continue
        answer = str(label).strip()
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(question, data_source=meta.data_source),
            "env_kwargs": {
                "ground_truth": answer,
                "question": _build_question_with_hint(question, data_source=meta.data_source),
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows
 
def _process_math_local(meta: DatasetMeta, split: str) -> List[dict]:
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)

    # Your local directory contains:
    # - math500.jsonl
    # - minerva_math.jsonl
    #
    # We map:
    # - lighteval/MATH -> math500.jsonl
    # - math_dapo     -> minerva_math.jsonl
    if meta.data_source in ("lighteval/MATH",):
        path = _find_first_existing_file(root, ["math500.jsonl", "math.jsonl", "math.json"])
    elif meta.data_source in ("math_dapo",):
        path = _find_first_existing_file(root, ["minerva_math.jsonl", "math_dapo.jsonl", "math_dapo.json"])
    else:
        # default to math500
        path = _find_first_existing_file(root, ["math500.jsonl", "minerva_math.jsonl"])

    if not path:
        return []

    # jsonl lines contain: problem/solution/answer/(unique_id)
    # We'll split by index (deterministic) using a 90/10 split by default.
    try:
        n = sum(1 for _ in _jsonl_iter(path))
    except Exception as e:
        logger.warning("Failed counting local math file %s: %s", path, e)
        return []

    if n == 0:
        return []
    cut = int(0.9 * n)

    rows: List[dict] = []
    for i, ex in enumerate(_jsonl_iter(path)):
        is_train = i < cut
        if split == "train" and not is_train:
            continue
        if split == "test" and is_train:
            continue

        question = ex.get("problem") or ex.get("question") or ""
        answer = ex.get("answer") or ex.get("solution") or ""
        if question == "" or answer == "":
            continue

        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(question, data_source=meta.data_source),
            "env_kwargs": {
                "ground_truth": answer,
                "question": _build_question_with_hint(question, data_source=meta.data_source),
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def _process_math_local_split_specific(meta: DatasetMeta, split: str) -> List[dict]:
    """Try to load math from local split-specific files (train/test).

    If only an unsplit raw jsonl exists (e.g. math500.jsonl), return [] so
    that we can prefer HF official splits when possible.
    """
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)

    parquet_candidates = [
        f"{split}.parquet",
        f"math_{split}.parquet",
        f"*{split}*.parquet",
    ]
    jsonl_candidates = [
        f"{split}.jsonl",
        f"math_{split}.jsonl",
        f"*{split}*.jsonl",
    ]

    selected_path: Optional[str] = None
    selected_kind: str = ""
    for pat in parquet_candidates:
        matches = glob.glob(os.path.join(root, pat))
        if matches:
            selected_path = matches[0]
            selected_kind = "parquet"
            break
    if selected_path is None:
        for pat in jsonl_candidates:
            matches = glob.glob(os.path.join(root, pat))
            if matches:
                selected_path = matches[0]
                selected_kind = "jsonl"
                break

    if not selected_path:
        return []

    rows: List[dict] = []
    if selected_kind == "parquet":
        df = pd.read_parquet(selected_path)
        q_col = next((c for c in ["problem", "question"] if c in df.columns), None)
        a_col = next((c for c in ["solution", "answer"] if c in df.columns), None)
        if q_col is None or a_col is None:
            return []
        for i, ex in enumerate(df.to_dict(orient="records")):
            question = ex.get(q_col) or ""
            answer = ex.get(a_col) or ""
            if question == "" or answer == "":
                continue
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question, data_source=meta.data_source),
                "env_kwargs": {
                    "ground_truth": answer,
                    "question": _build_question_with_hint(question, data_source=meta.data_source),
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                },
                "reward_model": {"ground_truth": answer},
                "extra_info": {"index": i, "split": split},
            })
        return rows

    # selected_kind == "jsonl"
    for i, ex in enumerate(_jsonl_iter(selected_path)):
        question = ex.get("problem") or ex.get("question") or ""
        answer = ex.get("solution") or ex.get("answer") or ""
        if question == "" or answer == "":
            continue
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(question, data_source=meta.data_source),
            "env_kwargs": {
                "ground_truth": answer,
                "question": _build_question_with_hint(question, data_source=meta.data_source),
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def _process_aime_local(meta: DatasetMeta, split: str) -> List[dict]:
    """Load AIME-like problems from local files.
    AIME is test-only: all samples go to the test set."""
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)

    data_source = meta.data_source
    parquet_candidates: List[str] = []
    if "2024" in data_source:
        parquet_candidates = ["aime24.parquet", "*aime*24*.parquet", "*aime*2024*.parquet"]
    elif "2025" in data_source:
        parquet_candidates = ["AIME25_v2.parquet", "*aime*25*.parquet", "*aime*2025*.parquet"]
    else:
        parquet_candidates = [f"*{data_source}*.parquet", "*aime*.parquet"]

    parquet_path: Optional[str] = None
    for pat in parquet_candidates:
        matches = glob.glob(os.path.join(root, pat))
        if matches:
            parquet_path = matches[0]
            break

    if parquet_path and parquet_path.endswith(".parquet"):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning("Failed reading local AIME parquet %s: %s", parquet_path, e)
        else:
            if df is not None and len(df) > 0:
                q_col = "problem" if "problem" in df.columns else ("question" if "question" in df.columns else None)
                gt_col = "solution" if "solution" in df.columns else ("answer" if "answer" in df.columns else None)
                if q_col is not None and gt_col is not None:
                    rows: List[dict] = []
                    for i, ex in enumerate(df.to_dict(orient="records")):
                        question = ex.get(q_col, "") or ""
                        answer = ex.get(gt_col, "") or ""
                        if question == "" or answer == "":
                            continue
                        rows.append({
                            "data_source": meta.data_source,
                            "task_category": meta.task_category,
                            "prompt": _build_prompt(question, data_source=meta.data_source),
                            "env_kwargs": {
                                "ground_truth": answer,
                                "question": _build_question_with_hint(question, data_source=meta.data_source),
                                "data_source": meta.data_source,
                                "task_category": meta.task_category,
                            },
                            "reward_model": {"ground_truth": answer},
                            "extra_info": {"index": i, "split": "test"},
                        })
                    return rows

    # Fallback: jsonl
    patterns: List[str] = []
    if "2024" in data_source:
        patterns = ["*aime*2024*.jsonl", "*aime*24*.jsonl", "*aime2024*"]
    elif "2025" in data_source:
        patterns = ["*aime*2025*.jsonl", "*aime*25*.jsonl", "*aime2025*"]
    else:
        patterns = [f"*{data_source}*"]

    path = None
    for pat in patterns:
        matches = glob.glob(os.path.join(root, pat))
        if matches:
            path = matches[0]
            break
    if not path:
        return []

    rows: List[dict] = []
    for i, ex in enumerate(_jsonl_iter(path)):
        question = ex.get("problem") or ex.get("question") or ""
        answer = ex.get("answer") or ex.get("solution") or ""
        if question == "" or answer == "":
            continue
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(question, data_source=meta.data_source),
            "env_kwargs": {
                "ground_truth": answer,
                "question": _build_question_with_hint(question, data_source=meta.data_source),
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": "test"},
        })
    return rows


# Datasets that only appear in the test set (too few samples / no train split).
TEST_ONLY_SOURCES = {"aime_2024", "aime_2025"}


def process_aime(meta: DatasetMeta, split: str) -> List[dict]:
    if split == "train":
        return []
    if LOCAL_DATA_DIR:
        return _process_aime_local(meta=meta, split=split)
    return []


def _process_squad_local(meta: DatasetMeta, split: str) -> List[dict]:
    """
    Load SQuAD-style data from LOCAL_DATA_DIR if present.

    We try a few common layouts:
    1) Separate parquet/jsonl for train/validation/test
    2) One parquet with an explicit split column

    Expected output matches the schema produced by process_squad().
    """
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)

    # Candidate parquet/jsonl filenames/patterns.
    # Your local artifacts may come from your preprocessing pipeline, so
    # we keep patterns broad.
    parquet_candidates = [
        "squad_train.parquet",
        "squad_validation.parquet",
        "squad_test.parquet",
        "*squad*train*.parquet",
        "*squad*validation*.parquet",
        "*squad*test*.parquet",
        "squad.parquet",
        "*squad*.parquet",
    ]
    jsonl_candidates = [
        "squad_train.jsonl",
        "squad_validation.jsonl",
        "squad_test.jsonl",
        "*squad*train*.jsonl",
        "*squad*validation*.jsonl",
        "*squad*test*.jsonl",
        "squad.jsonl",
        "*squad*.jsonl",
    ]

    target_splits = {"train": {"train"}, "test": {"validation", "test"}}
    keep_splits = target_splits.get(split, {split})

    def _extract_answers(ans_field) -> List[str]:
        if ans_field is None:
            return []
        # HF style: {"text": [...], "answer_start": [...]}
        if isinstance(ans_field, dict):
            txt = ans_field.get("text", [])
            if isinstance(txt, list):
                return [str(x) for x in txt if x is not None]
            if isinstance(txt, str):
                return [txt]
            return []
        if isinstance(ans_field, list):
            return [str(x) for x in ans_field if x is not None]
        if isinstance(ans_field, str):
            return [ans_field]
        return []

    # Prefer parquet when available.
    for pat in parquet_candidates:
        matches = glob.glob(os.path.join(root, pat))
        if not matches:
            continue
        path = matches[0]
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue

        # If there's an explicit split column, filter it.
        split_col = None
        for cand in ["split", "data_split", "set"]:
            if cand in df.columns:
                split_col = cand
                break
        if split_col is not None:
            mask = df[split_col].astype(str).isin(keep_splits)
            df = df[mask]
        if len(df) == 0:
            continue

        rows: List[dict] = []
        # Column heuristics.
        q_col = next((c for c in ["question", "question_text", "query"] if c in df.columns), None)
        c_col = next((c for c in ["context", "context_text", "paragraph"] if c in df.columns), None)
        ans_col = next((c for c in ["answers", "answer"] if c in df.columns), None)
        # Some local formats might store answer text directly.
        if q_col is None or c_col is None:
            continue
        if ans_col is None:
            # try singular 'answer'
            ans_col = "answers" if "answers" in df.columns else None
        if ans_col is None:
            continue

        for i, ex in enumerate(df.to_dict(orient="records")):
            question = ex.get(q_col, "") or ""
            context = ex.get(c_col, "") or ""
            answers = _extract_answers(ex.get(ans_col))
            if question == "" or context == "" or not answers:
                continue
            question_str = f"Context: {context}\n\nQuestion: {question}"
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question_str, data_source=meta.data_source),
                "env_kwargs": {
                    "ground_truth": answers,
                    "question": _build_question_with_hint(question_str, data_source=meta.data_source),
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                },
                "reward_model": {"ground_truth": {"target": answers}},
                "extra_info": {"index": i, "split": split},
            })
        if rows:
            return rows

    # Fallback to jsonl (line-delimited records).
    for pat in jsonl_candidates:
        matches = glob.glob(os.path.join(root, pat))
        if not matches:
            continue
        path = matches[0]
        rows: List[dict] = []
        i = 0
        try:
            for ex in _jsonl_iter(path):
                # split filtering if exists
                ex_split = ex.get("split", ex.get("set", None))
                if ex_split is not None and str(ex_split) not in keep_splits:
                    continue
                question = ex.get("question") or ex.get("question_text") or ""
                context = ex.get("context") or ex.get("context_text") or ""
                answers = _extract_answers(ex.get("answers") or ex.get("answer"))
                if question == "" or context == "" or not answers:
                    continue
                question_str = f"Context: {context}\n\nQuestion: {question}"
                rows.append({
                    "data_source": meta.data_source,
                    "task_category": meta.task_category,
                    "prompt": _build_prompt(question_str, data_source=meta.data_source),
                    "env_kwargs": {
                        "ground_truth": answers,
                        "question": _build_question_with_hint(question_str, data_source=meta.data_source),
                        "data_source": meta.data_source,
                        "task_category": meta.task_category,
                    },
                    "reward_model": {"ground_truth": {"target": answers}},
                    "extra_info": {"index": i, "split": split},
                })
                i += 1
        except Exception:
            continue
        if rows:
            return rows

    return []

def process_squad(meta: DatasetMeta, split: str) -> List[dict]:
    # Prefer local SQuAD artifacts if provided.
    if LOCAL_DATA_DIR:
        rows_local = _process_squad_local(meta=meta, split=split)
        if rows_local:
            return rows_local

    from datasets import load_dataset
    try:
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map[split])
    except Exception as e:
        logger.warning("Could not load %s from HF: %s. Returning empty.", meta.data_source, e)
        return []

    rows = []
    for i, ex in enumerate(ds):
        answers = list(set(ex["answers"]["text"]))
        question = f"Context: {ex['context']}\n\nQuestion: {ex['question']}"
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(question, data_source=meta.data_source),
            "env_kwargs": {"ground_truth": answers, "question": _build_question_with_hint(question, data_source=meta.data_source),
                           "data_source": meta.data_source, "task_category": meta.task_category},
            "reward_model": {"ground_truth": {"target": answers}},
            "extra_info": {"index": i, "split": split},
        })
    return rows

def process_simpleqa(meta: DatasetMeta, split: str) -> List[dict]:
    """OpenAI SimpleQA – attempt HF load; fall back to placeholder."""
    rows = []
    try:
        from datasets import load_dataset
        if split == "train":
            ds = []
        else:
            ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("problem", ex.get("question", ""))
            answer = ex.get("answer", ex.get("solution", ""))
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question, data_source=meta.data_source),
                "env_kwargs": {"ground_truth": answer, "question": _build_question_with_hint(question, data_source=meta.data_source),
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": {"target": [answer] if isinstance(answer, str) else answer}},
                "extra_info": {"index": i, "split": split},
            })
    except Exception as e:
        logger.warning("Could not load %s: %s. Creating placeholder.", meta.data_source, e)
    return rows


def process_search_r1(meta: DatasetMeta, split: str) -> List[dict]:
    # Align with original verl-agent usage: read preprocessed search parquet
    # from local train/test files instead of loading PeterJinGo/searchR1_*.
    if not LOCAL_DATA_DIR:
        logger.warning(
            "LOCAL_DATA_DIR is required for search_qa. Expected %s split parquet under TOKEN_AGENT_DATA_DIR/search_qa.",
            split,
        )
        return []

    root = os.path.join(LOCAL_DATA_DIR, "search_qa")
    parquet_file = "train.parquet" if split == "train" else "test.parquet"
    mixed_path = os.path.join(root, parquet_file)
    if not os.path.exists(mixed_path):
        logger.warning("search_qa parquet not found: %s", mixed_path)
        return []

    try:
        df = pd.read_parquet(mixed_path)
    except Exception as e:
        logger.warning("Failed reading search_qa parquet %s: %s", mixed_path, e)
        return []

    rows: List[dict] = []
    for i, rec in enumerate(df.to_dict(orient="records")):
        # Round-trip through JSON to convert all numpy/pandas scalar types
        # (numpy arrays, numpy int64, etc.) into plain Python objects, and
        # to ensure every complex field is a standard dict/list/str.
        try:
            rec = json.loads(json.dumps(rec, default=str))
        except Exception:
            continue

        prompt = rec.get("prompt")
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except Exception:
                prompt = _build_prompt(prompt, data_source=meta.data_source)
        if not isinstance(prompt, list):
            prompt = _build_prompt(str(prompt, data_source=meta.data_source) if prompt else "", data_source=meta.data_source)

        env_kwargs = rec.get("env_kwargs")
        if not isinstance(env_kwargs, dict):
            env_kwargs = {}

        # Normalise ground_truth inside env_kwargs to {"target": [...]} format,
        # which is what SearchEnv.compute_score expects (ground_truth["target"]).
        # The raw search_qa parquet stores ground_truth as a plain list or string.
        raw_gt = env_kwargs.get("ground_truth")
        if raw_gt is not None and not isinstance(raw_gt, dict):
            if isinstance(raw_gt, str):
                raw_gt = [raw_gt]
            elif not isinstance(raw_gt, list):
                raw_gt = [str(raw_gt)]
            env_kwargs["ground_truth"] = {"target": raw_gt}

        reward_model = rec.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = None

        extra_info = rec.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info.setdefault("index", i)
        extra_info.setdefault("split", split)

        normalised = {
            "data_source": rec.get("data_source", meta.data_source),
            "task_category": rec.get("task_category", meta.task_category),
            "prompt": prompt,
            "env_kwargs": env_kwargs,
            "reward_model": reward_model,
            "extra_info": extra_info,
        }
        _ensure_system_prompt(normalised)
        rows.append(normalised)
    return rows
    

def process_env_placeholder(meta: DatasetMeta, split: str) -> List[dict]:
    """
    For environment-based datasets (ALFWorld, WebShop, Sokoban, …) the
    actual data lives inside the env package. We produce a lightweight
    parquet row that carries the data_source and task_category so the
    MixedEnvironmentManager can dispatch correctly at runtime.
    """
    return [{
        "data_source": meta.data_source,
        "task_category": meta.task_category,
        "prompt": _build_prompt(f"[Environment task: {meta.data_source}]"),
        "env_kwargs": {"data_source": meta.data_source, "task_category": meta.task_category},
        "reward_model": None,
        "extra_info": {"index": 0, "split": split},
    }]


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_PROCESSORS: Dict[str, Callable] = {
    "openai/gsm8k": process_gsm8k,
    "lighteval/MATH": process_math,
    "math_dapo": process_math,
    "aime_2024": process_aime,
    "aime_2025": process_aime,
    "squad": process_squad,
    "simpleqa": process_simpleqa,
    "alfworld": process_env_placeholder,
    "webshop": process_env_placeholder,
    "sokoban": process_env_placeholder,
    "search_qa": process_search_r1,
    "ezpoints": process_env_placeholder,
}

for _ds in ["search_qa"]:
    _PROCESSORS[_ds] = process_search_r1

for _mm in ["mmstar", "sqa", "mmvet", "pope", "mmb", "math_v", "ai2d"]:
    _PROCESSORS[_mm] = process_env_placeholder


def process_dataset(meta: DatasetMeta, split: str) -> List[dict]:
    processor = _PROCESSORS.get(meta.data_source)
    if processor is None:
        logger.warning("No processor for %s – skipping", meta.data_source)
        return []
    try:
        return processor(meta, split)
    except Exception as e:
        # Make preprocessing resilient: one missing/unreachable dataset
        # should not crash the whole mixed-benchmark build.
        logger.warning("Processor failed for %s (%s): %s. Skipping.", meta.data_source, split, e)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Token-Agent mixed benchmark parquets.")
    parser.add_argument("--local_dir", default="~/data/token_agent_mixed",
                        help="Directory to save parquet files")
    parser.add_argument(
        "--local_dataset_dir",
        default=os.environ.get("TOKEN_AGENT_DATA_DIR", None),
        help="Local dataset directory (for gsm8k/math/aime when HF is unavailable). "
             "You can also set env TOKEN_AGENT_DATA_DIR.",
    )
    parser.add_argument("--active_categories", default="0,1,2,3",
                        help="Comma-separated category IDs to include (0-4)")
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Cap rows per dataset (for quick testing)")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    global LOCAL_DATA_DIR
    LOCAL_DATA_DIR = args.local_dataset_dir
    os.makedirs(local_dir, exist_ok=True)
    active_cats = set(int(c) for c in args.active_categories.split(","))

    try:
        sample_ratios: Dict[str, float] = json.loads(args.sample_ratios)
    except json.JSONDecodeError as e:
        raise ValueError(f"--sample_ratios must be a valid JSON dict, got: {args.sample_ratios!r}") from e

    import random
    random.seed(42)

    for split in ["train", "test"]:
        all_rows: List[dict] = []
        for meta in get_active_datasets():
            if meta.task_category not in active_cats:
                continue
            if meta.hf_repo_id is None and meta.data_source not in _PROCESSORS:
                logger.info("Skipping %s (no HF repo and no processor)", meta.data_source)
                continue

            logger.info("Processing %s / %s ...", meta.data_source, split)
            rows = process_dataset(meta, split)
            if args.max_per_dataset and len(rows) > args.max_per_dataset:
                rows = rows[: args.max_per_dataset]

            ratio = sample_ratios.get(meta.data_source)
            if ratio is not None:
                if not (0.0 < ratio <= 1.0):
                    raise ValueError(
                        f"sample_ratios[{meta.data_source!r}] must be in (0, 1], got {ratio}"
                    )
                sample_size = max(1, int(len(rows) * ratio))
                rows = random.sample(rows, sample_size)
                logger.info("  -> %d rows (sampled %.0f%% from original)", len(rows), ratio * 100)
            else:
                logger.info("  -> %d rows", len(rows))
            all_rows.extend(rows)

        if not all_rows:
            logger.warning("No rows for split=%s", split)
            continue
        for row in all_rows:
            _ensure_system_prompt(row)

        # Serialise env_kwargs and reward_model as JSON strings before writing
        # to parquet. These nested dicts have inconsistent internal schemas
        # across datasets (e.g. ground_truth is str in gsm8k but List[str] in
        # squad), which causes pyarrow struct-column schema conflicts.
        # RLHFDataset.__getitem__ deserialises them back with json.loads.
        serialised_rows = []
        for row in all_rows:
            serialised_row = dict(row)
            env_kwargs = serialised_row.get("env_kwargs")
            serialised_row["env_kwargs"] = (
                json.dumps(env_kwargs, default=str) if isinstance(env_kwargs, dict) else "{}"
            )
            reward_model = serialised_row.get("reward_model")
            serialised_row["reward_model"] = (
                json.dumps(reward_model, default=str) if isinstance(reward_model, dict) else None
            )
            serialised_rows.append(serialised_row)
        df = pd.DataFrame(serialised_rows)
        out_path = os.path.join(local_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
