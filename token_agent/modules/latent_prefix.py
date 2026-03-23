"""
Latent prefix extraction and triplet loss for Token-Agent.

The model generates ``<latent>...</latent>`` tokens as part of its response.
We extract the **hidden states** (not logits) of the tokens inside that block
and use them as each sample's latent representation.

- ``extract_latent_hidden_states`` : locate and pool hidden states
- ``compute_triplet_loss``         : batch-hard triplet loss
- ``compute_category_prefixes``    : per-category average
- ``CategoryPrefixTracker``        : EMA tracker across batches
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def find_latent_spans(
    token_ids: torch.Tensor,
    latent_start_id: int,
    latent_end_id: int,
) -> list[Tuple[int, int]]:
    """
    For a 1-D tensor of token IDs, return ``(start, end)`` index pairs
    marking tokens strictly *between* ``<latent>`` and ``</latent>``.

    Returns an empty list if no valid span is found.
    """
    ids = token_ids.tolist()
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] == latent_start_id:
            j = i + 1
            while j < len(ids) and ids[j] != latent_end_id:
                j += 1
            if j < len(ids):
                if j > i + 1:
                    spans.append((i + 1, j))
            i = j + 1
        else:
            i += 1
    return spans


def extract_latent_hidden_states(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    latent_start_id: int,
    latent_end_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract per-sample latent representations from model hidden states.

    Parameters
    ----------
    hidden_states : (batch, seq_len, hidden_dim)
        Last-layer hidden states from the model forward pass.
    token_ids : (batch, seq_len)
        Token IDs of the full sequence (prompt + response).
    latent_start_id / latent_end_id : int
        Token IDs for ``<latent>`` and ``</latent>``.

    Returns
    -------
    latent_repr : (batch, hidden_dim)
        Mean-pooled hidden states of the tokens inside ``<latent>...</latent>``
        for each sample.  Samples without a valid span get a zero vector.
    valid_mask : (batch,)
        Boolean mask – True where a valid ``<latent>`` span was found.
    """
    batch_size, _, hidden_dim = hidden_states.shape
    device = hidden_states.device

    latent_repr = torch.zeros(batch_size, hidden_dim, device=device)
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for b in range(batch_size):
        spans = find_latent_spans(token_ids[b], latent_start_id, latent_end_id)
        if not spans:
            continue
        start, end = spans[0]
        latent_repr[b] = hidden_states[b, start:end].mean(dim=0)
        valid_mask[b] = True

    return latent_repr, valid_mask


# ---------------------------------------------------------------------------
# Triplet loss  (batch-hard mining)
# ---------------------------------------------------------------------------

def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Squared L2 distances: (N, N)."""
    dot = embeddings @ embeddings.T
    sq_norm = torch.diag(dot)
    dist = sq_norm.unsqueeze(0) - 2.0 * dot + sq_norm.unsqueeze(1)
    return dist.clamp(min=0.0)


def compute_triplet_loss(
    latent_repr: torch.Tensor,
    category_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Batch-hard triplet loss on latent representations.

    For each anchor, we pick the *hardest positive* (farthest same-category)
    and *hardest negative* (closest different-category).

    Parameters
    ----------
    latent_repr : (batch, hidden_dim)
    category_ids : (batch,)   int tensor
    valid_mask : (batch,)     bool tensor (only valid samples participate)
    margin : float

    Returns
    -------
    loss : scalar tensor (0 if fewer than 2 categories present)
    """
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=latent_repr.device, requires_grad=True)

    emb = latent_repr[valid_mask]
    cats = category_ids[valid_mask]
    n = emb.shape[0]

    unique_cats = cats.unique()
    if len(unique_cats) < 2:
        return torch.tensor(0.0, device=latent_repr.device, requires_grad=True)

    dist = _pairwise_distances(emb)

    same_cat = cats.unsqueeze(0) == cats.unsqueeze(1)
    diff_cat = ~same_cat

    # hardest positive: max dist among same-category pairs
    pos_dist = dist * same_cat.float()
    hardest_pos, _ = pos_dist.max(dim=1)

    # hardest negative: min dist among diff-category pairs (mask with large value)
    large = dist.max().detach() + 1.0
    neg_dist = dist + (~diff_cat).float() * large
    hardest_neg, _ = neg_dist.min(dim=1)

    losses = F.relu(hardest_pos - hardest_neg + margin)
    return losses.mean()


# ---------------------------------------------------------------------------
# Category-level prefix computation
# ---------------------------------------------------------------------------

def compute_category_prefixes(
    latent_repr: torch.Tensor,
    category_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    Average latent representations within each category.

    Returns
    -------
    dict mapping ``category_id (int) -> mean_repr (hidden_dim,)``
    """
    result: Dict[int, torch.Tensor] = {}
    for cat in category_ids[valid_mask].unique().tolist():
        mask = valid_mask & (category_ids == cat)
        if mask.any():
            result[cat] = latent_repr[mask].mean(dim=0)
    return result


# ---------------------------------------------------------------------------
# EMA tracker for category prefixes across batches
# ---------------------------------------------------------------------------

class CategoryPrefixTracker:
    """
    Maintains an exponential moving average of category-level latent
    representations across training iterations.
    """

    def __init__(self, num_categories: int = 5, momentum: float = 0.9):
        self.num_categories = num_categories
        self.momentum = momentum
        self._ema: Dict[int, torch.Tensor] = {}

    def update(self, cat_prefixes: Dict[int, torch.Tensor]):
        for cat, vec in cat_prefixes.items():
            vec_detached = vec.detach()
            if cat not in self._ema:
                self._ema[cat] = vec_detached.clone()
            else:
                self._ema[cat] = (
                    self.momentum * self._ema[cat]
                    + (1 - self.momentum) * vec_detached
                )

    def get(self, cat: int) -> Optional[torch.Tensor]:
        return self._ema.get(cat)

    def state_dict(self) -> Dict[int, torch.Tensor]:
        return {k: v.clone() for k, v in self._ema.items()}

    def load_state_dict(self, d: Dict[int, torch.Tensor]):
        self._ema = {k: v.clone() for k, v in d.items()}
