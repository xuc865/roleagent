# Token-Agent

Token-Agent 是一种让同一个 Agent 在面对不同任务时自动适配不同推理路径的方法。核心思想是：通过模型生成的 `<latent>...</latent>` token 的 hidden states 作为 latent prefix，用 triplet loss 拉近同类任务、拉远不同类任务的表征，同时与 GRPO loss 联合训练。

## 目录结构

```
token_agent/
├── config/                          # Hydra 配置
│   ├── token_agent_trainer.yaml     # Token-Agent 训练配置
│   ├── grpo_baseline_trainer.yaml   # GRPO baseline 配置（无 latent prefix）
│   └── mixed_tool_config.yaml       # 空 tool config（满足 validate_config 断言）
├── data/
│   ├── dataset_registry.py          # 数据集注册表：data_source → task_category 映射
│   └── preprocess_mixed_benchmark.py # 数据预处理：下载并转换为统一 parquet
├── environments/
│   ├── single_turn_env.py           # 单轮任务环境包装器（数学、QA）
│   └── mixed_env_manager.py         # 混合环境路由：按 task_category 分发到子环境
├── modules/
│   └── latent_prefix.py             # latent hidden state 提取 + triplet loss + EMA tracker
├── prompts/
│   └── unified_system_prompt.py     # 统一系统 prompt（包含所有工具描述）
├── rewards/
│   ├── penalty_reward.py            # 过度推理惩罚 + 错误工具惩罚
│   └── episode_reward_manager.py    # 自定义 reward manager（在 episode reward 上叠加惩罚）
├── trainer/
│   ├── token_agent_actor.py         # TokenAgentActor：GRPO loss + triplet loss 联合训练
│   ├── fsdp_workers.py              # FSDP worker mixin：将标准 actor 替换为 TokenAgentActor
│   └── main_token_agent.py          # 训练入口：环境构建、tokenizer 注册、训练循环
├── analysis/
│   ├── failure_mode_analysis.py     # 失败模式分析器：mode collapse、过度推理、错误工具
│   └── collect_eval_records.py      # 端到端评估：模型推理 → reward 计算 → 失败模式报告
└── scripts/
    ├── preprocess_data.sh           # 数据预处理脚本
    ├── train_token_agent.sh         # Token-Agent 训练脚本
    ├── train_grpo_baseline.sh       # GRPO baseline 训练脚本
    └── analyze_baseline.sh          # 失败模式分析脚本
```

## 任务类别

| ID | 名称 | 数据集 | 思维模式 |
|----|------|--------|----------|
| 0 | math_reasoning | GSM8K, MATH, math_dapo, aime_2024*, aime_2025* | 需要长链推理 |
| 1 | quick_qa | SQuAD | 快问快答，无需深度思考 |
| 2 | direct_qa | SimpleQA, AA-Omniscience | 直接回答，无需搜索 |
| 3 | search_qa | NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle | 需要搜索工具 |
| 4 | action_env | ALFWorld, WebShop | 需要交互式动作 |
| 5 | game_env | Sokoban, EZPoints | （接口已实现，暂不混入训练） |
| 6 | multimodal | MMStar, SQA, MMVet, ... | （接口已实现，暂不混入训练） |

\* aime_2024、aime_2025 数据量少且无 train split，**仅进入测试集**。

### 训练集 / 测试集划分

| cat | 数据集 | 训练集 | 测试集 |
|-----|--------|--------|--------|
| 0 | openai/gsm8k | ✓ | ✓ |
| 0 | lighteval/MATH | ✓ | ✓ |
| 0 | math_dapo | ✓ | ✓ |
| 0 | aime_2024 | ✗ | ✓ |
| 0 | aime_2025 | ✗ | ✓ |
| 1 | squad | ✓ | ✓ |
| 2 | simpleqa | ✓ | ✓ |
| 2 | aa_omniscience | ✓ | ✓ |
| 3 | searchR1_nq/triviaqa/popqa/hotpotqa/2wiki/musique/bamboogle | ✓ | ✓ |
| 4 | alfworld | ✓（环境自带 train split） | ✓ |
| 4 | webshop | ✓（环境自带 train split） | ✓ |

**所有样本的 system prompt 统一使用 `UNIFIED_SYSTEM_PROMPT`（包含全部工具描述），训练和评估阶段一致。**

## 快速开始

### 1. 数据准备

```bash
# 预处理混合 benchmark 数据
# 默认下载 category 0-3 的数据并转为统一 parquet
bash token_agent/scripts/preprocess_data.sh

# 或手动控制参数
python -m token_agent.data.preprocess_mixed_benchmark \
    --local_dir ~/data/token_agent_mixed \
    --active_categories 0,1,2,3,4 \
    --max_per_dataset 5000
```

输出文件：
- `/mnt/workspace/wxc/roleagent/data/token_agent_mixed/train.parquet` — 训练集
- `/mnt/workspace/wxc/roleagent/data/token_agent_mixed/test.parquet` — 测试集

每行包含统一格式：`data_source`, `task_category`, `prompt`, `env_kwargs`, `reward_model`, `extra_info`。

### 2. 训练 GRPO Baseline

先训练 GRPO baseline，观察它在混合 benchmark 上的行为：

```bash
bash token_agent/scripts/train_grpo_baseline.sh Qwen/Qwen2.5-3B-Instruct 8

# 或者直接用 Hydra overrides
python -m token_agent.trainer.main_token_agent \
    --config-path ../config --config-name grpo_baseline_trainer \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    trainer.n_gpus_per_node=8
```

关键配置（`grpo_baseline_trainer.yaml`）：
- `algorithm.token_agent.enable: False` — 不使用 latent prefix
- `env.env_name: mixed` — 使用混合环境
- `algorithm.adv_estimator: grpo` — GRPO 算法

### 3. 训练 Token-Agent

```bash
# train_token_agent.sh 接受 Hydra overrides 而非位置参数
bash token_agent/scripts/train_token_agent.sh \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    trainer.n_gpus_per_node=8
```

关键配置（`token_agent_trainer.yaml`）：
- `algorithm.token_agent.enable: True` — 启用 latent prefix + triplet loss
- `algorithm.token_agent.triplet_coeff: 0.1` — triplet loss 系数
- `algorithm.token_agent.triplet_margin: 1.0` — triplet margin
- `algorithm.token_agent.overthinking_penalty: 1.0` — 过度推理惩罚
- `algorithm.token_agent.wrong_tool_penalty: 0.2` — 错误工具惩罚

### 4. 分析失败模式

评估已训练模型，检测 mode collapse 等问题：

```bash
# 方式一：从 checkpoint 做推理并分析
bash token_agent/scripts/analyze_baseline.sh ./checkpoints/grpo_step_500 ./analysis_grpo 1000

# 方式二：如果已有评估结果（JSONL 格式）
python -m token_agent.analysis.failure_mode_analysis \
    --log_dir ./results/eval_records.jsonl \
    --format jsonl \
    --output_dir ./analysis_output/
```

输出报告包含：
- **Collapse Score**（0-1）：模型推理风格是否坍缩为单一模式
- **Over-Reasoning**：quick_qa/direct_qa 中 `<think>` 块的使用率及 reward 影响
- **Under-Reasoning**：数学题中跳过推理的比例
- **Wrong Tool Usage**：每个类别错误使用其他类别工具的比例
- **Response Style Distribution**：各类别响应长度分布

## 核心设计

### 统一系统 Prompt

所有任务共享一个系统 prompt，描述了 `search`、`action`、`answer` 三类工具。**不含任何思维模式的提示**——模型必须自行判断采用何种推理策略。Prompt 中要求模型在回答前先生成 `<latent>...</latent>` token。

### Latent Prefix 机制

1. 模型为每个任务生成 `<latent>...</latent>` token
2. 提取这些 token 的 hidden states，mean-pool 得到单个向量
3. Triplet loss（batch-hard）拉近同类任务的 latent 表征，拉远不同类任务
4. 同类任务的 latent 平均值作为该类的 category prefix（EMA 更新）
5. Triplet loss + GRPO loss 联合优化

### 评估惩罚

- **过度推理惩罚**：对 quick_qa (1) / direct_qa (2)，如果模型产生了 `<think>` 块，reward 减去 1.0
- **错误工具惩罚**：如果模型调用了不属于当前任务类别的工具（如在数学题中调用 search），且 base reward ≥ 0.2，则减去 0.2

### 混合环境路由

`MixedEnvironmentManager` 根据每个样本的 `task_category` 将其路由到对应的子环境管理器：
- Category 0-2 → `SingleTurnEnvironmentManager`（1 步完成）
- Category 3 → `SearchEnvironmentManager`（多轮搜索）
- Category 4 → `AlfWorldEnvironmentManager` / `WebshopEnvironmentManager`（多轮交互）

### EVAL 记录格式

分析工具接受 JSONL 格式，每行一个 JSON 对象：

```json
{
  "data_source": "openai/gsm8k",
  "task_category": 0,
  "question": "...",
  "response": "...",
  "reward": 1.0,
  "ground_truth": "42"
}
```

## 对上游 verl-agent 的改动

仅修改了 2 个上游文件：

1. **`verl/utils/reward_score/__init__.py`** — 新增 `squad`, `simpleqa` 的 reward 计算分支
2. **`agent_system/environments/env_manager.py`** — 在 `make_envs()` 中添加 `"mixed"` 环境的路由

所有其他代码均在 `token_agent/` 下，不影响上游功能。

## 依赖

在原有 verl-agent 依赖之上，额外需要：
- `datasets`（HuggingFace datasets，用于数据下载）
- `pandas`（parquet 读写）
- `vllm`（可选，用于 `collect_eval_records.py` 的快速推理）
