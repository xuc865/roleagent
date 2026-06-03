#!/usr/bin/env bash
# GRPO Baseline Training on Mixed Benchmark (no Token-Agent latent prefix).
#
# Usage:
#   bash token_agent/scripts/train_grpo_baseline.sh [MODEL_PATH] [NUM_GPUS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${1:-/mnt/workspace/wxc/Agent/models/Qwen2.5-3B-Instruct}"
NUM_GPUS="${2:-8}"

train_data_size=32
val_data_size=64
group_size=4

TRAIN_DATA="/mnt/workspace/wxc/roleagent/data/token_agent_mixed/train.parquet"
VAL_DATA="/mnt/workspace/wxc/roleagent/data/token_agent_mixed/test.parquet"


############################## 开启所有的环境
INDEX_PATH="/mnt/workspace/wxc/roleagent/agent_system/environments/env_package/search/data/e5_Flat.index"
CORPUS_PATH="/mnt/workspace/wxc/roleagent/agent_system/environments/env_package/search/data/wiki-18.jsonl"
RETRIEVAL_SERVER_LOG="/tmp/retrieval_server.log"
HF_ENDPOINT=https://hf-mirror.com

source /mnt/workspace/wxc/roleagent/.setup_env_ret.sh 
export PYTHONPATH=/mnt/workspace/wxc/roleagent:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:-}"
 
fuser -k 8866/tcp 2>/dev/null || true
python3 /mnt/workspace/wxc/roleagent/examples/search/retriever/retrieval_server.py \
--index_path $INDEX_PATH \
--corpus_path $CORPUS_PATH \
--topk 3 \
--retriever_name e5 \
--retriever_model intfloat/e5-base-v2 \
--faiss_gpu \
--port 8866 > $RETRIEVAL_SERVER_LOG 2>&1 &
RETRIEVAL_SERVER_PID=$!
echo "Retrieval server started with PID: $RETRIEVAL_SERVER_PID"
 


source /mnt/workspace/wxc/roleagent/.setup_env.sh
export PYTHONPATH=/mnt/workspace/wxc/roleagent:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
# Re-export WANDB_API_KEY after source to prevent setup_env from overriding it.
export WANDB_API_KEY="${WANDB_API_KEY:-}"

##########################3 开启结束



# 在这里指定你的设置结果
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
VAL_DATA_SOURCES="${VAL_DATA_SOURCES:-}"


##########################3 开启结束


python -m token_agent.trainer.main_token_agent \
    --config-name grpo_baseline_trainer \
    \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    "data.val_data_sources_str='${VAL_DATA_SOURCES}'" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_data_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=token_agent/config/mixed_tool_config.yaml \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=False \
    algorithm.token_agent.enable=False \
    algorithm.token_agent.wrong_tool_penalty=0.2 \
    algorithm.token_agent.overthinking_penalty=1.0 \
    \
    reward_model.enable=False \
    reward_model.reward_manager=episode \
    \
    env.env_name=mixed \
    env.seed=42 \
    env.max_steps=10 \
    env.history_length=2 \
    env.resources_per_worker.num_cpus=0.1 \
    env.resources_per_worker.num_gpus=0 \
    env.rollout.n=$group_size \
    'env.mixed.active_categories=[0,1,2,3]' \
    env.search.log_requests=false \
    env.search.search_url='http://127.0.0.1:8866/retrieve' \
    env.search.topk=3 \
    env.search.timeout=60 \
    env.alfworld.eval_dataset=eval_in_distribution \
    env.webshop.use_small=True \
    env.webshop.human_goals=False \
    \
    trainer.project_name=token-agent \
    trainer.experiment_name=grpo_baseline_mixed \
    trainer.total_epochs=10 \
    trainer.balance_batch=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=999999 \
    trainer.val_before_train=False \
    'trainer.logger=["console","local_file"]' \
    "$@"
