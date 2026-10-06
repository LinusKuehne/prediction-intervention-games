from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .models import PredictorBundle
from .scm import AttackMechanisms, NonlinearStableBlanketSCM

AttackObjective = Literal["signed_error", "mse", "prediction"]


@dataclass
class AttackResult:
    objective_name: str
    bound: float
    cost: float | None
    best_objective_value: float
    best_regularized_value: float
    intervention_strength: float
    attacked_test_mse: float
    attack: AttackMechanisms


def _objective(y: Tensor, pred: Tensor, objective: AttackObjective) -> Tensor:
    if objective == "signed_error":
        return torch.mean(y - pred)
    if objective == "mse":
        return torch.mean((y - pred) ** 2)
    if objective == "prediction":
        return torch.mean(pred)
    raise ValueError(objective)


def _intervention_strength(delta_x1: Tensor, delta_x4: Tensor) -> Tensor:
    return torch.mean(delta_x1.square() + delta_x4.square())


def optimize_attack(
    *,
    scm: NonlinearStableBlanketSCM,
    predictor: PredictorBundle,
    bound: float,
    cost: float | None,
    intervene_on_x1: bool,
    simple: bool,
    x4_uses_x1_x3: bool,
    objective: AttackObjective,
    restarts: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    eval_size: int,
    seed: int,
    x4_uses_x1: bool = False,
    init_attack: AttackMechanisms | None = None,
    init_candidates: Sequence[AttackMechanisms] = (),
    depth: int = 2,
    slope: float = 1.0,
    lr_schedule: str = "constant",
    dropout: float = 0.0,
    output: str = "tanh",
    zero_init: bool = False,
    weight_decay: float = 0.0,
    bound_anneal_steps: int = 0,
) -> AttackResult:
    """Train the attack networks; keep the best of several starts by held-out value.

    ``restarts`` random starts (all initialised at ``init_attack`` if given) plus
    one start per network in ``init_candidates`` (e.g. the network trained for the
    previous bound: continuation in b).  ``depth``/``slope`` configure the
    networks; ``lr_schedule`` is "constant" or "cosine" (annealed to 0).
    ``dropout``/``output``/``zero_init`` are network options (see
    ``AttackNetwork``); ``weight_decay`` > 0 switches to AdamW.  With
    ``bound_anneal_steps`` > 0 the bound grows linearly from 0 to ``bound``
    over that many steps (the networks learn the sign pattern of the shifts
    while the shifts are small, which avoids collapsing to a constant sign).
    """
    # Predictor is frozen.
    predictor.model.eval()
    for p in predictor.model.parameters():
        p.requires_grad_(False)

    def make_attack() -> AttackMechanisms:
        return AttackMechanisms(
            bound=bound,
            hidden_dim=hidden_dim,
            intervene_on_x1=intervene_on_x1,
            simple=simple,
            x4_uses_x1_x3=x4_uses_x1_x3,
            x4_uses_x1=x4_uses_x1,
            depth=depth,
            slope=slope,
            dropout=dropout,
            output=output,
            zero_init=zero_init,
        ).to(scm.device)

    best_attack: AttackMechanisms | None = None
    best_value: float | None = None
    best_regularized_value: float | None = None

    inits: list[AttackMechanisms | None] = [init_attack] * restarts + list(
        init_candidates
    )
    for restart, init in enumerate(inits):
        torch.manual_seed(seed + 1000 * restart)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1000 * restart)
        attack = make_attack()
        if init is not None:  # warm start or continuation from another bound
            attack.load_state_dict(init.state_dict())
        optimizer = (
            torch.optim.AdamW(attack.parameters(), lr=lr, weight_decay=weight_decay)
            if weight_decay > 0
            else torch.optim.Adam(attack.parameters(), lr=lr)
        )
        attack.train()
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
            if lr_schedule == "cosine"
            else None
        )
        generator = torch.Generator(device=scm.device)
        generator.manual_seed(seed + 1000 * restart + 123)

        nets = [n for n in (attack.x1_net, attack.x4_net) if n is not None]
        for step in range(steps):
            if bound_anneal_steps > 0:
                for net in nets:
                    net.bound = bound * min(1.0, (step + 1) / bound_anneal_steps)
            optimizer.zero_grad(set_to_none=True)
            x_adv, y_adv, delta_x1, delta_x4 = scm.sample_with_intervention_info(
                batch_size, attack=attack, generator=generator
            )
            pred = predictor.model(predictor.select_and_standardize(x_adv))
            value = _objective(y_adv, pred, objective)
            strength = _intervention_strength(delta_x1, delta_x4)
            if cost is None:
                loss = value if objective != "mse" else -value
            elif objective != "mse":
                loss = value + cost * strength
            else:
                loss = -value + cost * strength
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        for net in nets:
            net.bound = float(bound)
        attack.eval()
        eval_gen = torch.Generator(device=scm.device)
        eval_gen.manual_seed(seed + 5000 + restart)
        with torch.no_grad():
            x_eval, y_eval, delta_x1_eval, delta_x4_eval = (
                scm.sample_with_intervention_info(
                    eval_size, attack=attack, generator=eval_gen
                )
            )
            pred_eval = predictor.model(predictor.select_and_standardize(x_eval))
            value_eval = _objective(y_eval, pred_eval, objective).item()
            strength_eval = _intervention_strength(delta_x1_eval, delta_x4_eval).item()

        regularized_value_eval = value_eval
        if cost is not None:
            regularized_value_eval = (
                value_eval - cost * strength_eval
                if objective == "mse"
                else value_eval + cost * strength_eval
            )

        better = best_value is None
        if objective != "mse" and best_regularized_value is not None:
            better = regularized_value_eval < best_regularized_value
        elif objective == "mse" and best_regularized_value is not None:
            better = regularized_value_eval > best_regularized_value

        if better:
            best_value = value_eval
            best_regularized_value = regularized_value_eval
            best_attack = make_attack()
            best_attack.load_state_dict(attack.state_dict())
            best_attack.eval()

    assert (
        best_attack is not None
        and best_value is not None
        and best_regularized_value is not None
    )
    test_gen = torch.Generator(device=scm.device)
    test_gen.manual_seed(seed + 99999)
    with torch.no_grad():
        x_test_adv, y_test_adv, delta_x1_test, delta_x4_test = (
            scm.sample_with_intervention_info(
                eval_size, attack=best_attack, generator=test_gen
            )
        )
        attacked_test_mse = torch.mean(
            (y_test_adv - predictor.predict(x_test_adv)) ** 2
        ).item()
        intervention_strength = _intervention_strength(
            delta_x1_test, delta_x4_test
        ).item()

    return AttackResult(
        objective_name=objective,
        bound=float(bound),
        cost=cost,
        best_objective_value=float(best_value),
        best_regularized_value=float(best_regularized_value),
        intervention_strength=float(intervention_strength),
        attacked_test_mse=float(attacked_test_mse),
        attack=best_attack,
    )
