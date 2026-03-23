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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOCAL_DATA_DIR: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-dataset processors: each returns a list[dict] of unified rows
# ---------------------------------------------------------------------------

def _build_prompt(question: str, system: str = "") -> List[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    return msgs


def process_gsm8k(meta: DatasetMeta, split: str) -> List[dict]:
    # Prefer local files if provided (HF may be unavailable in some environments).
    if LOCAL_DATA_DIR:
        rows_local = _process_gsm8k_local(meta=meta, split=split)
        if rows_local:
            return rows_local

    from datasets import load_dataset
    try:
        ds = load_dataset(meta.hf_repo_id, "main", split=meta.split_map[split])
    except Exception as e:
        logger.warning("Could not load %s from HF: %s. Skipping.", meta.data_source, e)
        return []

    rows = []
    for i, ex in enumerate(ds):
        answer = ex["answer"].split("####")[-1].strip()
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(ex["question"]),
            "env_kwargs": {"ground_truth": answer, "question": ex["question"],
                           "data_source": meta.data_source, "task_category": meta.task_category},
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def process_math(meta: DatasetMeta, split: str) -> List[dict]:
    # Prefer local math files if provided.
    if LOCAL_DATA_DIR:
        rows_local = _process_math_local(meta=meta, split=split)
        if rows_local:
            return rows_local

    from datasets import load_dataset
    try:
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map[split])
    except Exception as e:
        logger.warning(
            "Could not load %s (hf_repo_id=%s): %s. Skipping this dataset.",
            meta.data_source,
            meta.hf_repo_id,
            e,
        )
        return []

    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(ex["problem"]),
            "env_kwargs": {"ground_truth": ex["solution"], "question": ex["problem"],
                           "data_source": meta.data_source, "task_category": meta.task_category},
            "reward_model": {"ground_truth": ex["solution"]},
            "extra_info": {"index": i, "split": split},
        })
    return rows


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
            "prompt": _build_prompt(text),
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
            "prompt": _build_prompt(question),
            "env_kwargs": {
                "ground_truth": answer,
                "question": question,
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def _process_aime_local(meta: DatasetMeta, split: str) -> List[dict]:
    """Load AIME-like problems from local files if present."""
    assert LOCAL_DATA_DIR is not None
    root = _local_root_for_data_source(meta.data_source)

    # Prefer parquet (your local AIME files are parquet).
    # - aime24.parquet: columns = [id, problem, solution, url]
    # - AIME25_v2.parquet: columns = [id, problem, answer, year]
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
                n = len(df)
                cut = int(0.9 * n)
                q_col = "problem" if "problem" in df.columns else ("question" if "question" in df.columns else None)
                gt_col = "solution" if "solution" in df.columns else ("answer" if "answer" in df.columns else None)
                if q_col is not None and gt_col is not None:
                    rows: List[dict] = []
                    for i, ex in enumerate(df.to_dict(orient="records")):
                        is_train = i < cut
                        if split == "train" and not is_train:
                            continue
                        if split == "test" and is_train:
                            continue
                        question = ex.get(q_col, "") or ""
                        answer = ex.get(gt_col, "") or ""
                        if question == "" or answer == "":
                            continue
                        rows.append({
                            "data_source": meta.data_source,
                            "task_category": meta.task_category,
                            "prompt": _build_prompt(question),
                            "env_kwargs": {
                                "ground_truth": answer,
                                "question": question,
                                "data_source": meta.data_source,
                                "task_category": meta.task_category,
                            },
                            "reward_model": {"ground_truth": answer},
                            "extra_info": {"index": i, "split": split},
                        })
                    return rows

    # Try common filename patterns.
    data_source = meta.data_source  # e.g. aime_2024
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

    # Expect jsonl lines with problem/answer (or question/answer).
    try:
        n = sum(1 for _ in _jsonl_iter(path))
    except Exception as e:
        logger.warning("Failed counting local aime file %s: %s", path, e)
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
            "prompt": _build_prompt(question),
            "env_kwargs": {
                "ground_truth": answer,
                "question": question,
                "data_source": meta.data_source,
                "task_category": meta.task_category,
            },
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def process_aime(meta: DatasetMeta, split: str) -> List[dict]:
    # AIME is expected to be local-only in your setup.
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
                "prompt": _build_prompt(question_str),
                "env_kwargs": {
                    "ground_truth": answers,
                    "question": question_str,
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
                    "prompt": _build_prompt(question_str),
                    "env_kwargs": {
                        "ground_truth": answers,
                        "question": question_str,
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
            "prompt": _build_prompt(question),
            "env_kwargs": {"ground_truth": answers, "question": question,
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
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("problem", ex.get("question", ""))
            answer = ex.get("answer", ex.get("solution", ""))
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question),
                "env_kwargs": {"ground_truth": answer, "question": question,
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": {"target": [answer] if isinstance(answer, str) else answer}},
                "extra_info": {"index": i, "split": split},
            })
    except Exception as e:
        logger.warning("Could not load %s: %s. Creating placeholder.", meta.data_source, e)
    return rows


def process_aa_omniscience(meta: DatasetMeta, split: str) -> List[dict]:
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question),
                "env_kwargs": {"ground_truth": answer, "question": question,
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": {"target": [answer] if isinstance(answer, str) else answer}},
                "extra_info": {"index": i, "split": split},
            })
    except Exception as e:
        logger.warning("Could not load %s: %s. Creating placeholder.", meta.data_source, e)
    return rows


def process_search_r1(meta: DatasetMeta, split: str) -> List[dict]:
    """
    Search-R1 datasets are already preprocessed via the existing
    verl-agent pipeline. This function downloads the raw HF data and
    converts it into the unified schema.
    """
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            golden = ex.get("golden_answers", ex.get("answer", []))
            if isinstance(golden, str):
                golden = [golden]
            gt = {"target": golden}
            tools_kwargs = {
                "search": {
                    "create_kwargs": {
                        "ground_truth": gt,
                        "question": question,
                        "data_source": meta.data_source,
                    }
                }
            }
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question),
                "env_kwargs": {"ground_truth": gt, "question": question,
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": gt},
                "extra_info": {
                    "index": i,
                    "split": split,
                    "need_tools_kwargs": True,
                    "tools_kwargs": tools_kwargs,
                },
            })
    except Exception as e:
        logger.warning("Could not load %s: %s", meta.data_source, e)
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
    "aa_omniscience": process_aa_omniscience,
    "alfworld": process_env_placeholder,
    "webshop": process_env_placeholder,
    "sokoban": process_env_placeholder,
    "ezpoints": process_env_placeholder,
}

for _ds in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa",
            "searchR1_hotpotqa", "searchR1_2wikimultihopqa",
            "searchR1_musique", "searchR1_bamboogle"]:
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
            logger.info("  -> %d rows", len(rows))
            all_rows.extend(rows)

        if not all_rows:
            logger.warning("No rows for split=%s", split)
            continue

        df = pd.DataFrame(all_rows)
        out_path = os.path.join(local_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
