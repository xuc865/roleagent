"""
Training entry-point for Token-Agent.

Based on ``verl/trainer/main_ppo.py`` with the following additions:
- Uses ``MixedEnvironmentManager`` instead of a single-env ``make_envs()``
- Passes ``task_category`` through the data pipeline
- Registers ``<latent>`` / ``</latent>`` special tokens
- Wires up ``TokenAgentActor`` via a custom FSDP worker subclass
- Applies Token-Agent penalty rewards
"""

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


# ------------------------------------------------------------------
# Hydra entry
# ------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="token_agent_trainer", version_base=None)
def main(config):
    run_token_agent(config)


def run_token_agent(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TokenAgentRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TokenAgentRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # ---- Environment ------------------------------------------------
        ta_cfg = config.algorithm.get("token_agent", {})
        if ta_cfg.get("enable", True) and config.env.get("env_name", "") == "mixed":
            from token_agent.environments.mixed_env_manager import make_mixed_envs
            envs, val_envs = make_mixed_envs(config)
        else:
            from agent_system.environments import make_envs
            envs, val_envs = make_envs(config)

        # ---- Tokenizer / processor ---------------------------------------
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # ---- Register <latent> / </latent> special tokens ----------------
        LATENT_START = "<latent>"
        LATENT_END = "</latent>"
        new_tokens = []
        for tok in [LATENT_START, LATENT_END]:
            if tok not in tokenizer.get_vocab():
                new_tokens.append(tok)
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            print(f"Added special tokens: {new_tokens}")

        latent_start_id = tokenizer.convert_tokens_to_ids(LATENT_START)
        latent_end_id = tokenizer.convert_tokens_to_ids(LATENT_END)
        print(f"<latent> id={latent_start_id}, </latent> id={latent_end_id}")

        # Store in config for downstream use
        with OmegaConf.read_write(config):
            if "token_agent" not in config.algorithm:
                OmegaConf.update(config, "algorithm.token_agent", {})
            config.algorithm.token_agent.latent_start_id = int(latent_start_id)
            config.algorithm.token_agent.latent_end_id = int(latent_end_id)

        # ---- Pass token_agent config to worker level -----------------------
        with OmegaConf.read_write(config):
            if not config.actor_rollout_ref.get("token_agent"):
                OmegaConf.update(config, "actor_rollout_ref.token_agent",
                                 OmegaConf.to_container(config.algorithm.get("token_agent", {}), resolve=True))
            else:
                OmegaConf.merge(config.actor_rollout_ref.token_agent,
                                config.algorithm.get("token_agent", {}))

        # ---- Worker classes -----------------------------------------------
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import CriticWorker

            use_token_agent = ta_cfg.get("enable", True)
            if use_token_agent:
                from token_agent.trainer.fsdp_workers import (
                    TokenAgentActorRolloutRefWorker,
                    TokenAgentAsyncActorRolloutRefWorker,
                )
                actor_rollout_cls = (
                    TokenAgentAsyncActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else TokenAgentActorRolloutRefWorker
                )
            else:
                from verl.workers.fsdp_workers import (
                    ActorRolloutRefWorker,
                    AsyncActorRolloutRefWorker,
                )
                actor_rollout_cls = (
                    AsyncActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else ActorRolloutRefWorker
                )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )
            actor_rollout_cls = ActorRolloutRefWorker  # Token-Agent not yet supported on megatron
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker as _RefWorker
            role_worker_mapping[Role.RefPolicy] = ray.remote(_RefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # ---- Reward manager -----------------------------------------------
        use_token_agent = ta_cfg.get("enable", True) and config.env.get("env_name", "") == "mixed"
        if use_token_agent:
            from token_agent.rewards.episode_reward_manager import TokenAgentEpisodeRewardManager
            overthinking_pen = ta_cfg.get("overthinking_penalty", 1.0)
            wrong_tool_pen = ta_cfg.get("wrong_tool_penalty", 0.2)
            reward_fn = TokenAgentEpisodeRewardManager(
                tokenizer=tokenizer, num_examine=0, normalize_by_length=False,
                overthinking_penalty=overthinking_pen, wrong_tool_penalty=wrong_tool_pen,
            )
            val_reward_fn = TokenAgentEpisodeRewardManager(
                tokenizer=tokenizer, num_examine=1, normalize_by_length=False,
                overthinking_penalty=overthinking_pen, wrong_tool_penalty=wrong_tool_pen,
            )
        else:
            from agent_system.reward_manager import EpisodeRewardManager
            reward_fn = EpisodeRewardManager(
                tokenizer=tokenizer, num_examine=0, normalize_by_length=False
            )
            val_reward_fn = EpisodeRewardManager(
                tokenizer=tokenizer, num_examine=1, normalize_by_length=False
            )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl+env, keep n=1; GRPO grouping is via env.rollout.n"
        )

        # ---- Trajectory collector -----------------------------------------
        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(
            config=config, tokenizer=tokenizer, processor=processor
        )

        # ---- Dataset -------------------------------------------------------
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        train_dataset = _create_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_dataset = _create_dataset(
            config.data.val_files, config.data, tokenizer, processor
        )
        train_sampler = _create_sampler(config.data, train_dataset)

        # ---- Trainer -------------------------------------------------------
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _create_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"Custom dataset class '{data_config.custom_cls.name}' must inherit from Dataset"
            )
    else:
        dataset_cls = RLHFDataset

    return dataset_cls(
        data_files=data_paths, tokenizer=tokenizer, processor=processor, config=data_config
    )


def _create_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        g = torch.Generator()
        g.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=g)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
