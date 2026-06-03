"""
Collect evaluation records from a trained model checkpoint on the
mixed benchmark, then run failure mode analysis.

Usage::

    python -m token_agent.analysis.collect_eval_records \
        --model_path <checkpoint_path> \
        --data_path /mnt/workspace/wxc/roleagent/data/token_agent_mixedtest.parquet \
        --output_dir ./analysis_output/ \
        --max_samples 500 \
        --temperature 0.3

This script:
1. Loads the mixed benchmark test data
2. Runs inference with the specified model
3. Computes rewards (with penalties)
4. Feeds everything to FailureModeAnalyzer
5. Outputs a JSON report + console summary
"""

import argparse
import json
import logging
import os
from typing import List

import pandas as pd

from token_agent.analysis.failure_mode_analysis import (
    EvalRecord,
    FailureModeAnalyzer,
    _print_summary,
)
from token_agent.data.dataset_registry import get_task_category
from token_agent.rewards.penalty_reward import compute_reward_with_penalties

logger = logging.getLogger(__name__)


def load_test_data(data_path: str, max_samples: int = None) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df


def run_inference_vllm(
    model_path: str,
    prompts: List[str],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    top_p: float = 0.9,
) -> List[str]:
    """Run batch inference using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vllm is required for inference. Install: pip install vllm")

    llm = LLM(model=model_path, trust_remote_code=True)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def run_inference_hf(
    model_path: str,
    prompts: List[str],
    temperature: float = 0.3,
    max_new_tokens: int = 2048,
) -> List[str]:
    """Fallback: run batch inference using HuggingFace generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
    )

    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.9,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def build_prompt_strings(df: pd.DataFrame, tokenizer=None) -> List[str]:
    """Build final prompt strings from the DataFrame."""
    from token_agent.prompts.unified_system_prompt import UNIFIED_SYSTEM_PROMPT

    prompts = []
    for _, row in df.iterrows():
        messages = row.get("prompt", [])
        if isinstance(messages, str):
            messages = json.loads(messages)

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": UNIFIED_SYSTEM_PROMPT}] + messages

        if tokenizer is not None:
            prompt_str = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        else:
            parts = []
            for m in messages:
                parts.append(f"<|{m['role']}|>\n{m['content']}")
            parts.append("<|assistant|>\n")
            prompt_str = "\n".join(parts)
        prompts.append(prompt_str)
    return prompts


def build_eval_records(
    df: pd.DataFrame,
    responses: List[str],
    overthinking_penalty: float = 1.0,
    wrong_tool_penalty: float = 0.2,
) -> List[EvalRecord]:
    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(responses):
            break
        data_source = row.get("data_source", "unknown")
        task_category = int(row.get("task_category", get_task_category(data_source)))
        env_kwargs = row.get("env_kwargs", {})
        if isinstance(env_kwargs, str):
            env_kwargs = json.loads(env_kwargs)
        ground_truth = env_kwargs.get("ground_truth", "")

        reward = compute_reward_with_penalties(
            data_source=data_source,
            solution_str=responses[idx],
            ground_truth=ground_truth,
            task_category=task_category,
            overthinking_penalty=overthinking_penalty,
            wrong_tool_penalty=wrong_tool_penalty,
        )

        question = env_kwargs.get("question", "")
        records.append(EvalRecord(
            data_source=data_source,
            task_category=task_category,
            question=question,
            response=responses[idx],
            ground_truth=ground_truth,
            reward=reward,
        ))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the mixed benchmark and analyze failure modes."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/mnt/workspace/wxc/roleagent/data/token_agent_mixedtest.parquet")
    parser.add_argument("--output_dir", type=str, default="./analysis_output")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--overthinking_penalty", type=float, default=1.0)
    parser.add_argument("--wrong_tool_penalty", type=float, default=0.2)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test data from {data_path}...")
    df = load_test_data(data_path, args.max_samples)
    print(f"Loaded {len(df)} samples.")

    # Build prompts
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        tokenizer = None
    prompts = build_prompt_strings(df, tokenizer)

    print(f"Running inference with {args.backend}...")
    if args.backend == "vllm":
        responses = run_inference_vllm(args.model_path, prompts, args.temperature, args.max_tokens)
    else:
        responses = run_inference_hf(args.model_path, prompts, args.temperature, args.max_tokens)

    print("Computing rewards and building records...")
    records = build_eval_records(
        df, responses, args.overthinking_penalty, args.wrong_tool_penalty
    )

    # Save raw records
    records_path = os.path.join(args.output_dir, "eval_records.jsonl")
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps({
                "data_source": r.data_source,
                "task_category": r.task_category,
                "question": r.question[:500],
                "response": r.response,
                "reward": r.reward,
                "ground_truth": str(r.ground_truth)[:200] if r.ground_truth else "",
            }) + "\n")
    print(f"Raw records saved to {records_path}")

    # Run analysis
    analyzer = FailureModeAnalyzer(records)
    report = analyzer.full_report()

    report_path = os.path.join(args.output_dir, "failure_mode_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    _print_summary(report)


if __name__ == "__main__":
    main()
