# Role-Agent in `roleagent`

This document describes **Role-Agent–style** training support in this repo: **World-In-Agent (WIA)** and **Agent-In-World (AIW)**, aligned with the ideas in *Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution* (WIA + AIW), with implementation details that match our stack (string-based predicate, weighted sampling).

## What is enabled in code

| Component | Behavior | Where |
|-----------|----------|--------|
| **WIA** | Optional `<predict_next>` in the agent prompt; after each env step, step reward is scaled by `sigmoid(SequenceMatcher(prediction, next observation text))`. | `algorithm.role_agent.enable_wia`, `role_agent/wia_utils.py`, `agent_system/multi_turn_rollout/rollout_loop.py`, prompts via `agent_system/environments/env_manager.py` |
| **AIW** | On failed episodes, a short prompt fingerprint is stored; training indices are reweighted so similar past failures and the failed task are sampled more often. | `algorithm.role_agent.enable_aiw`, `role_agent/aiw_curriculum.py`, `verl/trainer/ppo/ray_trainer.py`, `TrajectoryCollector.multi_turn_loop` |

Hydra example:

`algorithm.role_agent.enable_wia=true algorithm.role_agent.enable_aiw=true`

Config keys under `algorithm.role_agent` in `verl/trainer/config/ppo_trainer.yaml`:

- `aiw_top_k`, `aiw_boost`, `aiw_self_boost`, `aiw_max_history`
- `aiw_similarity_thresh` — optional gate on **cross-task** AIW boosts only (same semantics as GiGPO’s similarity threshold for grouping: only pairs with ratio ≥ threshold contribute). `0.0` disables the gate.
- `text_match_max_chars` — if `> 0`, both strings are truncated to this length before `SequenceMatcher` for WIA and for AIW fingerprint matching. `0` means full strings (GiGPO-style parity; longest CPU cost).

## GiGPO alignment (non–main-innovation)

- **String score**: WIA and AIW use the same raw `difflib.SequenceMatcher(None, a, b).ratio()` as `gigpo.core_gigpo.text_similarity_ratio` / `are_similar`. The logic lives in `role_agent/wia_utils.py` so rollout workers do **not** import `gigpo.core_gigpo` (that module pulls in `torch`).
- **Discounting**: Example launch scripts set `algorithm.gamma=0.95` where the GiGPO recipes do (ALFWorld / WebShop PPO scripts, search GiGPO script).
- **Train loader**: With `enable_aiw=true`, use `data.dataloader_num_workers=0` in job scripts so the mutable weighted sampler stays well-defined (mirrors common Ray + dataloader practice for in-place weight updates).

## Performance collapse risks (design)

1. **Missing `<predict_next>`** — similarity is treated as `0`, so `sigmoid(0)=0.5` scales every step reward. That is a strong, systematic shrink unless the model reliably emits the tag.
2. **AIW weights** — large `aiw_boost` / `aiw_self_boost` can concentrate sampling on a few indices and starve others (distribution shift / collapse).
3. **Long observations** — full-string `SequenceMatcher` is \(O(n^2)\) in the worst case. Without `text_match_max_chars`, very long text anchors can dominate step time. The rollout path skips WIA when there is no truncation and `len(actual) > 8192`; set `text_match_max_chars` (e.g. `4096`) to compare a prefix instead of skipping.

## Throughput levers

- Set `algorithm.role_agent.text_match_max_chars` to a modest cap (e.g. `2048`–`8192`) if observations are long.
- Keep `data.dataloader_num_workers=0` when AIW is on (scripts under `examples/role_agent_trainer/` already do).
- Usual stack knobs: rollout micro-batches, `tensor_model_parallel_size`, `gpu_memory_utilization`, env `resources_per_worker.num_cpus`.

## Bash entrypoints (remote `/mnt` defaults)

Scripts live in `examples/role_agent_trainer/`. They `cd` to the repo root from their own location. Override data roots if your cluster layout differs:

- `VERL_DATA_ROOT` (default `/mnt/data/verl-agent`) for ALFWorld / WebShop parquet paths.
- `SEARCH_DATA_ROOT` (default `/mnt/data/searchR1_processed_direct`) for search.

## Compared to the paper

| Topic | Paper | This repository |
|-------|--------|------------------|
| GiGPO-style grouping | — | Unchanged: `gigpo/core_gigpo.py` |
| WIA predicate | Multi-step horizon, periodic LLM judge | One-step `<predict_next>` vs next observation text; **no** separate judge model |
| AIW | LLM failure analysis + LLM retrieval over a library | Fingerprint string similarity + **mutable weighted** train sampler |

## Other tooling

`token_agent/analysis/failure_mode_analysis.py` analyzes **mixed-benchmark evaluation** logs (mode collapse, tools, etc.). It is **not** the same code path as training-time `algorithm.role_agent.enable_aiw` (Parquet / `RayPPOTrainer` / `AIWCurriculum`).

## Optional extensions

1. **WIA** — Multi-horizon predictions, batched scoring with `role_agent/paper_prompts.PROMPT_COMPARE_PREDICTED_VS_ACTUAL`, or other judges.  
2. **AIW** — LLM failure summaries + `PROMPT_RETRIEVE_SIMILAR_FAILURES` over `role_agent/aiw_offline.py`-style entries.  

## Code references

- PPO / GiGPO training: `verl/trainer/ppo/ray_trainer.py`  
- Multi-turn rewards: `agent_system/multi_turn_rollout/rollout_loop.py`  
- Prompts for optional LLM workflows: `role_agent/paper_prompts.py`  
- Example jobs: `examples/role_agent_trainer/run_alfworld.sh`, `run_webshop.sh` (PPO), `run_webshop_gigpo.sh` (GiGPO), `run_search.sh` — see `examples/role_agent_trainer/README.md`
