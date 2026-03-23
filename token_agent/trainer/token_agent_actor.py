"""
TokenAgentActor – extends ``DataParallelPPOActor`` with:

1. Hidden-state extraction from ``<latent>...</latent>`` tokens
2. Triplet loss on latent representations (same-category close, diff far)
3. Combined loss: ``policy_loss + triplet_coeff * triplet_loss``

The model generates latent tokens as part of its normal response; we simply
read their hidden states and add an auxiliary objective.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor

from token_agent.modules.latent_prefix import (
    extract_latent_hidden_states,
    compute_triplet_loss,
    compute_category_prefixes,
    CategoryPrefixTracker,
)

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TokenAgentActor(DataParallelPPOActor):
    """
    Actor that adds latent-prefix triplet loss to the standard PPO/GRPO
    policy gradient objective.
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        latent_start_id: int = -1,
        latent_end_id: int = -1,
        triplet_coeff: float = 0.1,
        triplet_margin: float = 1.0,
        num_categories: int = 5,
        prefix_ema_momentum: float = 0.9,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.latent_start_id = latent_start_id
        self.latent_end_id = latent_end_id
        self.triplet_coeff = triplet_coeff
        self.triplet_margin = triplet_margin
        self.prefix_tracker = CategoryPrefixTracker(
            num_categories=num_categories,
            momentum=prefix_ema_momentum,
        )

    # ------------------------------------------------------------------
    # Forward with hidden-state extraction
    # ------------------------------------------------------------------

    def _forward_micro_batch_with_hidden(
        self,
        micro_batch: dict,
        temperature: float,
        calculate_entropy: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Same as ``_forward_micro_batch`` but also returns the last-layer
        hidden states so we can extract latent representations.

        Returns
        -------
        entropy    : (bs, response_len) or None
        log_probs  : (bs, response_len)
        hidden_states : (bs, seq_len, hidden_dim) or None
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inp[key] for inp in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=True,
            )

            logits = output.logits
            logits = logits / temperature

            last_hidden = output.hidden_states[-1]  # (bs, seqlen, hidden_dim)

            logits_resp = logits[:, -response_length - 1 : -1, :]
            log_probs = logprobs_from_logits(logits_resp, micro_batch["responses"])

            entropy = None
            if calculate_entropy:
                entropy = verl_F.entropy_from_logits(logits_resp)

        return entropy, log_probs, last_hidden

    # ------------------------------------------------------------------
    # update_policy with triplet loss
    # ------------------------------------------------------------------

    @GPUMemoryLogger(role="token_agent actor", logger=logger)
    def update_policy(self, data: DataProto) -> Dict:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # category_ids may be in non_tensor_batch
        category_ids_np = data.non_tensor_batch.get("task_category", None)

        if has_multi_modal:
            num_mini = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics: Dict = {}

        for epoch in range(self.config.ppo_epochs):
            mini_idx = 0
            for batch_idx, mini_data in enumerate(dataloader):
                if has_multi_modal:
                    mini_batch = mini_data
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_data.select(select_keys, non_tensor_select_keys).chunk(num_micro)
                elif self.config.use_dynamic_bsz:
                    max_tl = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_data, max_token_len=max_tl)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                all_latent_repr = []
                all_cat_ids = []

                for micro_data in micro_batches:
                    if isinstance(micro_data, DataProto):
                        micro_data = {**micro_data.batch.to(get_torch_device().current_device()),
                                      **micro_data.non_tensor_batch}
                    else:
                        micro_data = micro_data.to(get_torch_device().current_device())

                    responses = micro_data["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_data["attention_mask"]

                    if multi_turn:
                        response_mask = micro_data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = micro_data["old_log_probs"]
                    advantages = micro_data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob, hidden_states = self._forward_micro_batch_with_hidden(
                        micro_batch=micro_data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    # --- Policy loss (standard GRPO / PPO) ---
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    policy_loss = pg_loss
                    if calculate_entropy:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()

                    # --- Latent prefix extraction + triplet loss ---
                    triplet_loss = torch.tensor(0.0, device=policy_loss.device)
                    if (
                        self.latent_start_id >= 0
                        and self.latent_end_id >= 0
                        and hidden_states is not None
                    ):
                        full_ids = micro_data["input_ids"]
                        latent_repr, valid_mask = extract_latent_hidden_states(
                            hidden_states=hidden_states,
                            token_ids=full_ids,
                            latent_start_id=self.latent_start_id,
                            latent_end_id=self.latent_end_id,
                        )

                        micro_bs = full_ids.size(0)
                        start_idx = mini_idx
                        end_idx = mini_idx + micro_bs
                        mini_idx = end_idx

                        if category_ids_np is not None and end_idx <= len(category_ids_np):
                            cat_slice = category_ids_np[start_idx:end_idx]
                            cat_tensor = torch.tensor(
                                cat_slice.astype(int), device=full_ids.device
                            )
                        else:
                            cat_tensor = torch.zeros(micro_bs, dtype=torch.long, device=full_ids.device)

                        if valid_mask.any():
                            triplet_loss = compute_triplet_loss(
                                latent_repr=latent_repr,
                                category_ids=cat_tensor,
                                valid_mask=valid_mask,
                                margin=self.triplet_margin,
                            )
                            all_latent_repr.append(latent_repr[valid_mask].detach())
                            all_cat_ids.append(cat_tensor[valid_mask].detach())

                    # --- Combined loss ---
                    total_loss = policy_loss + self.triplet_coeff * triplet_loss

                    if self.config.use_dynamic_bsz:
                        loss = total_loss * (len(micro_data["input_ids"]) / self.config.ppo_mini_batch_size)
                    else:
                        loss = total_loss / self.gradient_accumulation
                    loss.backward()

                    step_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/triplet_loss": triplet_loss.detach().item(),
                    }
                    append_to_dict(metrics, step_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

                # Update category prefix tracker
                if all_latent_repr:
                    combined_repr = torch.cat(all_latent_repr, dim=0)
                    combined_cats = torch.cat(all_cat_ids, dim=0)
                    valid_all = torch.ones(combined_repr.size(0), dtype=torch.bool, device=combined_repr.device)
                    cat_pfx = compute_category_prefixes(combined_repr, combined_cats, valid_all)
                    self.prefix_tracker.update(cat_pfx)

        self.actor_optimizer.zero_grad()
        return metrics
