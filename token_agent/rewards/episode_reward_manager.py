"""
Token-Agent episode reward manager.

Extends the base ``EpisodeRewardManager`` to apply Token-Agent-specific
penalties (over-reasoning, wrong-tool) on top of the environment rewards.
"""

import numpy as np
import torch
from verl import DataProto

from agent_system.reward_manager.episode import EpisodeRewardManager
from token_agent.data.dataset_registry import get_task_category
from token_agent.rewards.penalty_reward import (
    _detect_wrong_tool,
    _has_think_block,
)


class TokenAgentEpisodeRewardManager(EpisodeRewardManager):

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length=False,
        overthinking_penalty: float = 1.0,
        wrong_tool_penalty: float = 0.2,
    ):
        super().__init__(tokenizer, num_examine, normalize_by_length)
        self.overthinking_penalty = overthinking_penalty
        self.wrong_tool_penalty = wrong_tool_penalty

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch["data_source"]
            task_category = get_task_category(data_source)

            episode_rewards = data_item.non_tensor_batch["episode_rewards"]
            episode_lengths = data_item.non_tensor_batch["episode_lengths"]

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards

            score = float(score)

            if task_category in (1, 2) and _has_think_block(response_str):
                score = max(0.0, score - self.overthinking_penalty)

            if score >= self.wrong_tool_penalty and _detect_wrong_tool(response_str, task_category):
                score = max(0.0, score - self.wrong_tool_penalty)

            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor
