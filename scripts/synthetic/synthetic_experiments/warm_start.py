"""Warm start of the neural population follower at the individuals' decision rules.

Used to check that a gap between the neural follower and the exact individual
follower is optimization error: the networks are first fitted by least squares
to the individuals' shifts (delta_1, delta_4) and then trained as in the paper.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .individual_follower import Noises
from .scm import AttackMechanisms


def fit_attack_to_rules(
    scm,
    noises: Noises,
    delta1: Tensor,
    delta4: Tensor,
    bound: float,
    *,
    hidden_dim: int = 64,
    x4_uses_x1: bool = True,
    steps: int = 3000,
    lr: float = 5e-3,
    seed: int = 0,
) -> AttackMechanisms:
    """Least-squares fit of the attack networks to per-individual shifts.

    The X1-network is fitted on (x2, eps1) -> delta1 and the X4-network on
    (y, x2, eps4[, x1]) -> delta4, where x1 and y are the individuals' realized
    values under delta1.  Full-batch Adam; the targets lie in [-b, b], which
    the networks' b * tanh output can represent.
    """
    torch.manual_seed(seed)
    attack = AttackMechanisms(
        bound=bound,
        hidden_dim=hidden_dim,
        intervene_on_x1=scm.intervene_on_x1,
        x4_uses_x1=x4_uses_x1,
    ).to(scm.device)
    x1 = scm._clean_x1(noises.x2, noises.eps1) + delta1
    y = scm._clean_y(x1, noises.x2, noises.eps_y)
    x3 = scm._clean_x3(y, noises.eps3)
    optimizer = torch.optim.Adam(attack.parameters(), lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            d4 = attack.perturb_x4(y, noises.x2, noises.eps4, x1=x1, x3=x3)
            loss = torch.mean((d4 - delta4) ** 2)
            if scm.intervene_on_x1:
                d1 = attack.perturb_x1(noises.x2, noises.eps1)
                loss = loss + torch.mean((d1 - delta1) ** 2)
            loss.backward()
            optimizer.step()
    return attack
