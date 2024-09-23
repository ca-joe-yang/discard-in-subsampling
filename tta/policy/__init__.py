from .base import BasePolicy
from .standard import StandardPolicy
from .expanded import ExpandedPolicy

def get_ttamodule(cfg, dm) -> BasePolicy:
    policy = cfg.TTA.POLICY
    match policy:
        case 'standard':
            return StandardPolicy(dm)
        case 'expanded':
            return ExpandedPolicy(dm)
        case _:
            raise ValueError(policy)