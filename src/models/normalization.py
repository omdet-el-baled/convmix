from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def valid_num_groups(channels, requested_groups=8):
    if channels < 1 or requested_groups < 1:
        raise ValueError("GroupNorm dimensions must be positive")
    return next(g for g in range(min(channels, requested_groups), 0, -1) if channels % g == 0)



def adaptive_group_norm(channels, groups=8):
    """Factory returns GroupNorm itself, retaining existing state-dict keys."""
    return nn.GroupNorm(valid_num_groups(channels, groups), channels)
