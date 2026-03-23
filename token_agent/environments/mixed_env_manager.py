"""
MixedEnvironmentManager – dispatches each sample in a batch to the correct
sub-environment based on its ``task_category``.

Single-turn tasks (math, QA) finish after 1 step; multi-turn tasks
(search, action envs) continue for multiple steps.  The existing
``vanilla_multi_turn_loop`` already tracks per-env ``is_done``, so mixed
step-counts work natively.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

from token_agent.data.dataset_registry import get_task_category


class MixedEnvironmentManager:
    """
    Routes each sample to the correct sub-EnvironmentManager based on
    ``task_category`` carried in the per-sample kwargs / batch metadata.

    Instantiated with a dict of sub-managers keyed by task_category.
    """

    def __init__(
        self,
        sub_managers: Dict[int, Any],
        config,
    ):
        self.sub_managers = sub_managers
        self.config = config

        self._batch_size: int = 0
        self._cat_ids: np.ndarray = np.array([], dtype=int)
        self._cat_to_indices: Dict[int, List[int]] = {}
        self._cat_to_local: Dict[int, Dict[int, int]] = {}

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        ``kwargs`` is either a list[dict] (one per sample) with a
        ``task_category`` key, or ``None`` for env-only datasets.
        """
        if kwargs is None:
            kwargs = [{}] * self._batch_size

        self._batch_size = len(kwargs)
        self._cat_ids = np.array(
            [kw.get("task_category", 0) for kw in kwargs], dtype=int
        )

        self._cat_to_indices = defaultdict(list)
        for i, c in enumerate(self._cat_ids):
            self._cat_to_indices[c].append(i)
        self._cat_to_local = {
            c: {gi: li for li, gi in enumerate(idxs)}
            for c, idxs in self._cat_to_indices.items()
        }

        all_obs_text: List[Optional[str]] = [None] * self._batch_size
        all_obs_image: List[Any] = [None] * self._batch_size
        all_obs_anchor: List[Any] = [None] * self._batch_size
        all_infos: List[Dict] = [{} for _ in range(self._batch_size)]

        for cat, indices in self._cat_to_indices.items():
            mgr = self.sub_managers.get(cat)
            if mgr is None:
                continue
            sub_kwargs = [kwargs[i] for i in indices]
            sub_obs, sub_infos = mgr.reset(sub_kwargs)

            sub_texts = sub_obs.get("text") or [None] * len(indices)
            sub_images = sub_obs.get("image")
            sub_anchors = sub_obs.get("anchor") or [None] * len(indices)

            for li, gi in enumerate(indices):
                all_obs_text[gi] = sub_texts[li] if li < len(sub_texts) else None
                if sub_images is not None:
                    all_obs_image[gi] = sub_images[li]
                all_obs_anchor[gi] = sub_anchors[li] if li < len(sub_anchors) else None
                all_infos[gi] = sub_infos[li] if li < len(sub_infos) else {}

        has_any_image = any(img is not None for img in all_obs_image)
        observations = {
            "text": all_obs_text,
            "image": all_obs_image if has_any_image else None,
            "anchor": all_obs_anchor,
        }
        return observations, all_infos

    def step(self, text_actions: List[str]):
        all_obs_text = [None] * self._batch_size
        all_obs_image = [None] * self._batch_size
        all_obs_anchor = [None] * self._batch_size
        all_rewards = np.zeros(self._batch_size, dtype=np.float32)
        all_dones = np.ones(self._batch_size, dtype=bool)
        all_infos: List[Dict] = [{"is_action_valid": True} for _ in range(self._batch_size)]

        for cat, indices in self._cat_to_indices.items():
            mgr = self.sub_managers.get(cat)
            if mgr is None:
                continue
            sub_actions = [text_actions[i] for i in indices]
            sub_obs, sub_rewards, sub_dones, sub_infos = mgr.step(sub_actions)

            sub_texts = sub_obs.get("text") or [None] * len(indices)
            sub_images = sub_obs.get("image")
            sub_anchors = sub_obs.get("anchor") or [None] * len(indices)

            for li, gi in enumerate(indices):
                all_obs_text[gi] = sub_texts[li] if li < len(sub_texts) else None
                if sub_images is not None:
                    all_obs_image[gi] = sub_images[li]
                all_obs_anchor[gi] = sub_anchors[li] if li < len(sub_anchors) else None
                all_rewards[gi] = float(sub_rewards[li])
                all_dones[gi] = bool(sub_dones[li])
                all_infos[gi] = sub_infos[li] if li < len(sub_infos) else {}

        has_any_image = any(img is not None for img in all_obs_image)
        next_observations = {
            "text": all_obs_text,
            "image": all_obs_image if has_any_image else None,
            "anchor": all_obs_anchor,
        }
        return next_observations, all_rewards, all_dones, all_infos

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def success_evaluator(self, total_infos, total_batch_list, episode_rewards, episode_lengths):
        success: Dict[str, list] = defaultdict(list)
        batch_size = len(total_batch_list)
        for batch_idx in range(batch_size):
            for i in reversed(range(len(total_batch_list[batch_idx]))):
                item = total_batch_list[batch_idx][i]
                if item["active_masks"]:
                    info = total_infos[batch_idx][i]
                    won = float(info.get("won", False))
                    success["success_rate"].append(won)
                    ds = info.get("data_source", "unknown")
                    success[f"{ds}_success_rate"].append(won)
                    break
        return {k: np.array(v) for k, v in success.items()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_mixed_envs(config):
    """
    Build a ``MixedEnvironmentManager`` containing sub-managers for each
    active task category, controlled by ``config.env.mixed.*``.
    """
    from token_agent.environments.single_turn_env import (
        SingleTurnEnvWrapper,
        SingleTurnEnvironmentManager,
    )
    from token_agent.rewards.penalty_reward import compute_reward_with_penalties

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(
        config.env.resources_per_worker, resolve=True
    )

    ta_cfg = config.algorithm.get("token_agent", {})
    overthinking_penalty = ta_cfg.get("overthinking_penalty", 1.0)
    wrong_tool_penalty = ta_cfg.get("wrong_tool_penalty", 0.2)

    def _single_turn_reward_fn(data_source, solution_str, ground_truth, **kwargs):
        task_category = kwargs.get("task_category")
        return compute_reward_with_penalties(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            task_category=task_category,
            overthinking_penalty=overthinking_penalty,
            wrong_tool_penalty=wrong_tool_penalty,
        )

    sub_train: Dict[int, Any] = {}
    sub_val: Dict[int, Any] = {}

    mixed_cfg = config.env.get("mixed", {})
    active_cats = set(mixed_cfg.get("active_categories", [0, 1, 2, 3, 4]))

    # --- Single-turn categories: 0 (math), 1 (quick_qa), 2 (direct_qa) ---
    for cat in [0, 1, 2]:
        if cat not in active_cats:
            continue
        train_envs = SingleTurnEnvWrapper(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            reward_fn=_single_turn_reward_fn,
        )
        val_envs_cat = SingleTurnEnvWrapper(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            reward_fn=_single_turn_reward_fn,
        )
        sub_train[cat] = SingleTurnEnvironmentManager(train_envs, config)
        sub_val[cat] = SingleTurnEnvironmentManager(val_envs_cat, config)

    # --- Search QA (category 3) ---
    if 3 in active_cats:
        try:
            from agent_system.environments.env_package.search import (
                build_search_envs,
                search_projection,
            )
            from agent_system.environments.env_manager import SearchEnvironmentManager

            _t = build_search_envs(
                seed=config.env.seed,
                env_num=config.data.train_batch_size,
                group_n=group_n,
                is_train=True,
                env_config=config.env,
            )
            _v = build_search_envs(
                seed=config.env.seed + 1000,
                env_num=config.data.val_batch_size,
                group_n=1,
                is_train=False,
                env_config=config.env,
            )
            pf = partial(search_projection)
            sub_train[3] = SearchEnvironmentManager(_t, pf, config)
            sub_val[3] = SearchEnvironmentManager(_v, pf, config)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Search env not available: %s", e)

    # --- Action Env (category 4): ALFWorld + WebShop ---
    if 4 in active_cats:
        import os
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        # -- ALFWorld --
        try:
            from agent_system.environments.env_package.alfworld import (
                build_alfworld_envs,
                alfworld_projection,
            )
            from agent_system.environments.env_manager import AlfWorldEnvironmentManager

            alf_cfg = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
            )
            if os.path.exists(alf_cfg):
                alf_env_kwargs = {"eval_dataset": config.env.get("alfworld", {}).get("eval_dataset", "eval_in_distribution")}
                _t = build_alfworld_envs(alf_cfg, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=alf_env_kwargs, resources_per_worker=resources_per_worker)
                _v = build_alfworld_envs(alf_cfg, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=alf_env_kwargs, resources_per_worker=resources_per_worker)
                pf = partial(alfworld_projection)
                sub_train[4] = AlfWorldEnvironmentManager(_t, pf, config)
                sub_val[4] = AlfWorldEnvironmentManager(_v, pf, config)
        except Exception as e:
            _logger.warning("ALFWorld env not available: %s", e)

        # -- WebShop --
        try:
            from agent_system.environments.env_package.webshop import (
                build_webshop_envs,
                webshop_projection,
            )
            from agent_system.environments.env_manager import WebshopEnvironmentManager

            ws_cfg = config.env.get("webshop", {})
            if ws_cfg.get("use_small", True):
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json")
                attr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json")
            else:
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../agent_system/environments/env_package/webshop/webshop/data/items_shuffle.json")
                attr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../agent_system/environments/env_package/webshop/webshop/data/items_ins_v2.json")
            ws_env_kwargs = {
                "observation_mode": "text",
                "num_products": None,
                "human_goals": ws_cfg.get("human_goals", True),
                "file_path": file_path,
                "attr_path": attr_path,
            }
            _wt = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=ws_env_kwargs, resources_per_worker=resources_per_worker)
            _wv = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=ws_env_kwargs, resources_per_worker=resources_per_worker)
            wpf = partial(webshop_projection)
            if 4 not in sub_train:
                sub_train[4] = WebshopEnvironmentManager(_wt, wpf, config)
                sub_val[4] = WebshopEnvironmentManager(_wv, wpf, config)
            else:
                _logger.info("ALFWorld already occupies cat 4; WebShop will share the same category via separate dispatch.")
        except Exception as e:
            _logger.warning("WebShop env not available: %s", e)

    envs = MixedEnvironmentManager(sub_train, config)
    val_envs = MixedEnvironmentManager(sub_val, config)
    return envs, val_envs
