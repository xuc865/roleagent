"""
Single-turn environment wrapper for tasks that require only one model response
(math reasoning, quick QA, direct QA).

The model generates a single response; we compute a reward and mark done=True.
This lets single-turn tasks participate in the same multi-turn rollout loop
used by interactive environments.
"""

from typing import Any, Dict, List, Tuple

import gym
import numpy as np


class SingleTurnEnvWrapper(gym.Env):
    """
    Vectorised 1-step environment for non-interactive tasks.

    On ``reset`` the question text becomes the observation.
    On ``step`` the response is scored against the ground truth and the
    episode terminates immediately.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        reward_fn=None,
    ):
        super().__init__()
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.reward_fn = reward_fn
        self._rng = np.random.RandomState(seed)

        self._questions: List[str] = []
        self._ground_truths: List[Any] = []
        self._data_sources: List[str] = []
        self._task_categories: List[int] = []

    def reset(self, kwargs: List[Dict]) -> Tuple[List[str], List[Dict]]:
        pad_n = self.batch_size - len(kwargs)
        dummy = {"ground_truth": "", "question": "", "data_source": "unknown"}
        padded = list(kwargs) + [dummy] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        self._questions = [kw["question"] for kw in padded]
        self._ground_truths = [kw["ground_truth"] for kw in padded]
        self._data_sources = [kw.get("data_source", "unknown") for kw in padded]
        self._task_categories = [kw.get("task_category", 0) for kw in padded]
        self._valid_mask = valid_mask

        obs = [q for q, v in zip(self._questions, valid_mask) if v]
        infos = [{"data_source": ds} for ds, v in zip(self._data_sources, valid_mask) if v]
        return obs, infos

    def step(self, actions: List[str]):
        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, (action, valid) in enumerate(zip(padded_actions, valid_mask)):
            if not valid:
                continue
            reward = 0.0
            if self.reward_fn is not None:
                reward = self.reward_fn(
                    data_source=self._data_sources[i],
                    solution_str=action,
                    ground_truth=self._ground_truths[i],
                    task_category=self._task_categories[i],
                )
            obs_list.append("")
            reward_list.append(float(reward))
            done_list.append(True)
            info_list.append({
                "data_source": self._data_sources[i],
                "won": reward >= 1.0,
            })

        return obs_list, reward_list, done_list, info_list

    def close(self):
        pass


class SingleTurnEnvironmentManager:
    """
    Minimal environment manager for single-turn tasks, following the
    same interface as ``EnvironmentManagerBase``.
    """

    def __init__(self, envs: SingleTurnEnvWrapper, config):
        self.envs = envs
        self.config = config

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        observations = {
            "text": obs,
            "image": None,
            "anchor": obs[:],
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        obs, rewards, dones, infos = self.envs.step(text_actions)

        for i, info in enumerate(infos):
            info["is_action_valid"] = True

        next_observations = {
            "text": obs,
            "image": None,
            "anchor": obs[:],
        }
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)
        return next_observations, rewards, dones, infos

    def success_evaluator(self, total_infos, total_batch_list, episode_rewards, episode_lengths):
        from collections import defaultdict
        success = defaultdict(list)
        for batch_idx in range(len(total_batch_list)):
            for i in reversed(range(len(total_batch_list[batch_idx]))):
                batch_item = total_batch_list[batch_idx][i]
                if batch_item["active_masks"]:
                    info = total_infos[batch_idx][i]
                    won = float(info.get("won", False))
                    success["success_rate"].append(won)
                    ds = info.get("data_source", "unknown")
                    success[f"{ds}_success_rate"].append(won)
                    break
        return {k: np.array(v) for k, v in success.items()}


def build_single_turn_envs(
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool,
    reward_fn=None,
):
    return SingleTurnEnvWrapper(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        reward_fn=reward_fn,
    )
