#!/usr/bin/env bash
# GRPO Training on ALFWorld with Token-Agent.
#
# Usage:
#   bash token_agent/scripts/train_grpo_alfworld.sh [MODEL_PATH] [NUM_GPUS]

set -uo pipefail
# Note: -e is intentionally omitted to prevent set -e from exiting on the harmless
# ResourceTracker cleanup error emitted by multiprocess on Python 3.12 shutdown.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${1:-/mnt/workspace/wxc/Agent/models/Qwen2.5-3B-Instruct}"
NUM_GPUS="${2:-8}"

train_data_size=16
val_data_size=128
group_size=8

# Only source setup_env if not already sourced by a parent script (e.g. sen.sh).
# This prevents double-sourcing which can reset environment variables set by the caller.
if [ -z "${VERL_ENV_SOURCED:-}" ]; then
    source /mnt/workspace/wxc/roleagent/.setup_env.sh
    export VERL_ENV_SOURCED=1
fi
export PYTHONPATH=/mnt/workspace/wxc/roleagent:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
# ALFWorld requires this data path; sen.sh sets it, but set a default here for standalone runs.
export ALFWORLD_DATA="${ALFWORLD_DATA:-/mnt/workspace/wxc/legacy/http/dat/alfworld_data}"
export HYDRA_FULL_ERROR=1

# Prepare ALFWorld data (text modality)
# Use || true to prevent set -e from exiting on the harmless ResourceTracker
# cleanup error that multiprocess emits on interpreter shutdown (Python 3.12 bug).
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size || true

TRAIN_DATA="$HOME/data/verl-agent/text/train.parquet"
VAL_DATA="$HOME/data/verl-agent/text/test.parquet"

python -m token_agent.trainer.main_token_agent \
    --config-name grpo_baseline_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_data_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=token_agent/config/mixed_tool_config.yaml \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=False \
    algorithm.token_agent.enable=True \
    algorithm.token_agent.wrong_tool_penalty=0.2 \
    algorithm.token_agent.overthinking_penalty=1.0 \
    \
    reward_model.enable=False \
    reward_model.reward_manager=episode \
    \
    "+ray_init.runtime_env.env_vars.ALFWORLD_DATA=$ALFWORLD_DATA" \
    \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.resources_per_worker.num_cpus=0.1 \
    env.resources_per_worker.num_gpus=0 \
    env.rollout.n=$group_size \
    env.alfworld.eval_dataset=eval_in_distribution \
    \
    trainer.critic_warmup=0 \
    trainer.project_name=token-agent \
    trainer.experiment_name=grpo_alfworld_token_agent \
    trainer.total_epochs=150 \
    trainer.balance_batch=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    'trainer.logger=["console","wandb"]' \
    "$@"
