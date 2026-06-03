"""
Custom FSDP worker that replaces the standard actor with TokenAgentActor
after model initialization.
"""

from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, register


class _TokenAgentActorMixin:
    """
    Mixin that overrides ``init_model`` to swap the standard
    ``DataParallelPPOActor`` with ``TokenAgentActor`` when the
    ``token_agent`` config section is present.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        ta_cfg = self.config.get("token_agent", {})
        if not ta_cfg.get("enable", False):
            return

        if not hasattr(self, "actor") or self.actor is None:
            return

        from token_agent.trainer.token_agent_actor import TokenAgentActor

        old_actor = self.actor
        self.actor = TokenAgentActor(
            config=old_actor.config,
            actor_module=old_actor.actor_module,
            actor_optimizer=old_actor.actor_optimizer,
            latent_start_id=int(ta_cfg.get("latent_start_id", -1)),
            latent_end_id=int(ta_cfg.get("latent_end_id", -1)),
            triplet_coeff=float(ta_cfg.get("triplet_coeff", 0.1)),
            triplet_margin=float(ta_cfg.get("triplet_margin", 1.0)),
            num_categories=int(ta_cfg.get("num_categories", 5)),
            prefix_ema_momentum=float(ta_cfg.get("prefix_ema_momentum", 0.9)),
        )


class TokenAgentActorRolloutRefWorker(_TokenAgentActorMixin, ActorRolloutRefWorker):
    pass


class TokenAgentAsyncActorRolloutRefWorker(_TokenAgentActorMixin, AsyncActorRolloutRefWorker):
    pass
