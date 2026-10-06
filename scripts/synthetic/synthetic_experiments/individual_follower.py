"""Individual followers for the synthetic prediction-intervention game.

The paper's synthetic follower (``adversary.py``) is a *population* follower:
one pair of intervention functions (delta_1, delta_4), parameterized by neural
networks, is optimized for the population objective E[k(f, X, Y)].

Section 2.1 of the paper argues that a population follower with action set
A_j = O \\ PA(j) is equivalent, from the leader's perspective, to many
*individual* followers who each observe their own x_O and choose their own
intervention values.  This module implements such individual followers
directly, with **no function class**: every individual solves their own
optimization problem by grid search, and expectations over not-yet-realized
exogenous noise are computed by Monte Carlo.

Because CH(E) = {X1, X4} and X1 -> Y -> X4, the individual decision is
sequential (the one-shot setup of Section 2.1 with O subset of ND(CH(E)) does
not apply literally).  We implement the perfect-recall sequential follower:

    stage 1: observe (x2, eps1)                 -> choose delta_1 in [-b, b]
    stage 2: observe (x1, x2, y, x3, eps4)      -> choose delta_4 in [-b, b]

Stage 2 integrates over eps5; stage 1 integrates over (eps_Y, eps3, eps4) and
the optimal stage-2 continuation (backward induction).  The stage-2
observation set is the maximal one allowed by Lemma 4 (A_4 = {X1, X3}) and
corresponds to the population follower with ``x4_uses_x1_x3=True``.

An *omniscient* individual follower, who knows all of their own exogenous
noises and picks (delta_1, delta_4) jointly for their realized payoff, is
provided as an upper bound; it lies outside the paper's framework (its
intervention depends on eps_Y, i.e., on noise that is not independent of Y).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .adversary import AttackObjective, _intervention_strength, _objective
from .models import PredictorBundle
from .scm import (
    AttackMechanisms,
    LinearGaussianStableBlanketSCM,
    NonlinearStableBlanketSCM,
)

SCM = NonlinearStableBlanketSCM | LinearGaussianStableBlanketSCM

# Column indices of X4 and X5 in the covariate vector (x1, x2, x3, x4, x5).
_X4_X5_COLUMNS = {3, 4}


@dataclass
class Noises:
    """Exogenous noise draws for a fixed set of individuals; each is (n, 1)."""

    x2: Tensor
    eps1: Tensor
    eps_y: Tensor
    eps3: Tensor
    eps4: Tensor
    eps5: Tensor

    @property
    def n(self) -> int:
        return int(self.x2.shape[0])


@dataclass
class IndividualFollowerConfig:
    grid_size_x1: int = 25  # grid points for delta_1 in [-b, b]
    grid_size_x4: int = 25  # grid points for delta_4 in [-b, b]
    mc_stage1: int = 32  # MC draws of (eps_Y, eps3, eps4) per individual, stage 1
    mc_stage2: int = 6  # MC draws of eps5 per stage-2 state
    omniscient_grid_size: int = 41  # per-axis grid for the omniscient follower
    chunk_individuals: int = 64  # individuals per chunk in the nested stage-1 loop
    predict_max_rows: int = 1 << 18  # rows per MLP forward call (memory cap)


@dataclass
class FollowerOutcome:
    follower: str
    attacked_mse: float
    follower_objective: float  # same sign convention as adversary._objective
    intervention_strength: float
    delta_x1: Tensor
    delta_x4: Tensor
    x: Tensor
    y: Tensor


def draw_noises(scm: SCM, n: int, generator: torch.Generator | None) -> Noises:
    """Draw noises in the same order as ``scm.sample_with_intervention_info``."""
    if scm.include_x6:
        raise NotImplementedError("Individual followers do not support X6.")
    x2 = scm._noise(n, generator)
    eps1 = scm._noise(n, generator)
    eps_y = scm._noise(n, generator)
    eps3 = scm._noise(n, generator)
    eps4 = scm._noise(n, generator)
    eps5 = scm._noise(n, generator)
    scm._noise(n, generator)  # eps6 is drawn by the SCM but unused here
    return Noises(x2=x2, eps1=eps1, eps_y=eps_y, eps3=eps3, eps4=eps4, eps5=eps5)


def follower_payoff(y: Tensor, pred: Tensor, objective: AttackObjective) -> Tensor:
    """Per-sample payoff the follower wants to *maximize* (cf. adversary._objective)."""
    if objective == "mse":
        return (y - pred) ** 2
    if objective == "prediction":
        return -pred
    if objective == "signed_error":
        return -(y - pred)
    raise ValueError(objective)


def _stage1_forward(
    scm: SCM, x2: Tensor, eps1: Tensor, eps_y: Tensor, eps3: Tensor, delta1: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    x1 = scm._clean_x1(x2, eps1) + delta1
    y = scm._clean_y(x1, x2, eps_y)
    x3 = scm._clean_x3(y, eps3)
    return x1, y, x3


def _stage2_forward(
    scm: SCM, y: Tensor, x2: Tensor, eps4: Tensor, eps5: Tensor, delta4: Tensor
) -> tuple[Tensor, Tensor]:
    x4 = scm._clean_x4_from_parents(y, x2, eps4) + delta4
    x5 = scm._clean_x5(x4, eps5)
    return x4, x5


def _assemble_x(x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor, x5: Tensor) -> Tensor:
    shape = torch.broadcast_shapes(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
    return torch.cat([t.expand(shape) for t in (x1, x2, x3, x4, x5)], dim=-1)


@torch.no_grad()
def _predict(predictor: PredictorBundle, x_full: Tensor, max_rows: int) -> Tensor:
    """Apply the (frozen) predictor to x_full of shape (..., 5); returns (..., 1)."""
    lead = x_full.shape[:-1]
    flat = x_full.reshape(-1, x_full.shape[-1])
    outputs = [
        predictor.model(predictor.select_and_standardize(chunk))
        for chunk in flat.split(max_rows, dim=0)
    ]
    return torch.cat(outputs, dim=0).reshape(*lead, 1)


def uses_x4_or_x5(predictor: PredictorBundle) -> bool:
    """If False, the follower's payoff does not depend on delta_4 at all."""
    return bool(set(predictor.subset_indices) & _X4_X5_COLUMNS)


@torch.no_grad()
def forward_with_attack(
    scm: SCM, noises: Noises, attack: AttackMechanisms | None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Replay the SCM with a (neural) population follower on fixed noise draws.

    Mirrors ``scm.sample_with_intervention_info`` exactly, but on given noises,
    so that different followers can be compared on the same individuals.
    """
    delta1 = torch.zeros_like(noises.x2)
    if attack is not None and scm.intervene_on_x1:
        delta1 = attack.perturb_x1(noises.x2, noises.eps1)
    x1, y, x3 = _stage1_forward(
        scm, noises.x2, noises.eps1, noises.eps_y, noises.eps3, delta1
    )
    delta4 = torch.zeros_like(noises.x2)
    if attack is not None:
        delta4 = attack.perturb_x4(y, noises.x2, noises.eps4, x1=x1, x3=x3)
    x4, x5 = _stage2_forward(scm, y, noises.x2, noises.eps4, noises.eps5, delta4)
    return _assemble_x(x1, noises.x2, x3, x4, x5), y, delta1, delta4


@torch.no_grad()
def _stage2_best(
    scm: SCM,
    predictor: PredictorBundle,
    objective: AttackObjective,
    *,
    x1: Tensor,
    x2: Tensor,
    y: Tensor,
    x3: Tensor,
    eps4: Tensor,
    grid4: Tensor,
    eps5_draws: Tensor,
    max_rows: int,
) -> tuple[Tensor, Tensor]:
    """Per-state optimal delta_4 and value.

    Inputs x1, x2, y, x3, eps4 are (B, 1); grid4 is (G,); eps5_draws is (B, M, 1).
    Returns (delta4_star (B, 1), value (B, 1)) where value = max_g mean_m payoff.
    """
    batch = x1.shape[0]
    n_grid = grid4.numel()
    n_mc = eps5_draws.shape[1]
    d4 = grid4.view(1, n_grid, 1, 1)
    e5 = eps5_draws.view(batch, 1, n_mc, 1)
    yb = y.view(batch, 1, 1, 1)
    x2b = x2.view(batch, 1, 1, 1)
    x4, x5 = _stage2_forward(scm, yb, x2b, eps4.view(batch, 1, 1, 1), e5, d4)
    x_full = _assemble_x(x1.view(batch, 1, 1, 1), x2b, x3.view(batch, 1, 1, 1), x4, x5)
    pred = _predict(predictor, x_full, max_rows)
    payoff = follower_payoff(yb, pred, objective).mean(dim=2).squeeze(-1)  # (B, G)
    value, idx = payoff.max(dim=1)
    return grid4[idx].view(batch, 1), value.view(batch, 1)


@torch.no_grad()
def _stage1_best(
    scm: SCM,
    predictor: PredictorBundle,
    objective: AttackObjective,
    *,
    x2: Tensor,
    eps1: Tensor,
    grid1: Tensor,
    grid4: Tensor,
    cfg: IndividualFollowerConfig,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Per-individual optimal delta_1 by backward induction (nested MC + grid)."""
    n = x2.shape[0]
    n_grid1 = grid1.numel()
    m1 = cfg.mc_stage1
    needs_stage2 = uses_x4_or_x5(predictor)
    delta1_star = torch.zeros(n, 1, device=x2.device)
    value_star = torch.zeros(n, 1, device=x2.device)

    def draw(*shape: int) -> Tensor:
        return torch.randn(*shape, device=x2.device, generator=generator)

    for start in range(0, n, cfg.chunk_individuals):
        stop = min(start + cfg.chunk_individuals, n)
        n_c = stop - start
        x2c = x2[start:stop].view(n_c, 1, 1, 1)
        eps1c = eps1[start:stop].view(n_c, 1, 1, 1)
        d1 = grid1.view(1, n_grid1, 1, 1)
        # Stage-1 MC noise, per individual, shared across the delta_1 grid (CRN).
        eps_y = draw(n_c, 1, m1, 1)
        eps3 = draw(n_c, 1, m1, 1)
        eps4 = draw(n_c, 1, m1, 1)
        x1, y, x3 = _stage1_forward(scm, x2c, eps1c, eps_y, eps3, d1)
        shape = (n_c, n_grid1, m1, 1)
        x1, x2e, eps4e = x1.expand(shape), x2c.expand(shape), eps4.expand(shape)
        if needs_stage2:
            batch = n_c * n_grid1 * m1
            _, v2 = _stage2_best(
                scm,
                predictor,
                objective,
                x1=x1.reshape(batch, 1),
                x2=x2e.reshape(batch, 1),
                y=y.reshape(batch, 1),
                x3=x3.reshape(batch, 1),
                eps4=eps4e.reshape(batch, 1),
                grid4=grid4,
                eps5_draws=draw(batch, cfg.mc_stage2, 1),
                max_rows=cfg.predict_max_rows,
            )
            v2 = v2.view(n_c, n_grid1, m1)
        else:
            # Payoff does not depend on (x4, x5): skip stage 2 (exact shortcut).
            placeholder = torch.zeros(shape, device=x2.device)
            x_full = _assemble_x(x1, x2e, x3, placeholder, placeholder)
            pred = _predict(predictor, x_full, cfg.predict_max_rows)
            v2 = follower_payoff(y, pred, objective).squeeze(-1)
        v1 = v2.mean(dim=2)  # (n_c, G1)
        value, idx = v1.max(dim=1)
        delta1_star[start:stop] = grid1[idx].view(n_c, 1)
        value_star[start:stop] = value.view(n_c, 1)
    return delta1_star, value_star


def _symmetric_grid(bound: float, size: int, device: torch.device) -> Tensor:
    return torch.linspace(-bound, bound, size, device=device)


def _outcome(
    name: str,
    predictor: PredictorBundle,
    objective: AttackObjective,
    x: Tensor,
    y: Tensor,
    delta1: Tensor,
    delta4: Tensor,
) -> FollowerOutcome:
    pred = predictor.predict(x)
    return FollowerOutcome(
        follower=name,
        attacked_mse=float(torch.mean((y - pred) ** 2)),
        follower_objective=float(_objective(y, pred, objective)),
        intervention_strength=float(_intervention_strength(delta1, delta4)),
        delta_x1=delta1,
        delta_x4=delta4,
        x=x,
        y=y,
    )


@torch.no_grad()
def population_follower_outcome(
    scm: SCM,
    predictor: PredictorBundle,
    objective: AttackObjective,
    noises: Noises,
    attack: AttackMechanisms | None,
    name: str = "population_nn",
) -> FollowerOutcome:
    x, y, delta1, delta4 = forward_with_attack(scm, noises, attack)
    return _outcome(name, predictor, objective, x, y, delta1, delta4)


@torch.no_grad()
def sequential_individual_follower(
    scm: SCM,
    predictor: PredictorBundle,
    objective: AttackObjective,
    noises: Noises,
    bound: float,
    cfg: IndividualFollowerConfig,
    seed: int,
    name: str = "individual_sequential",
) -> FollowerOutcome:
    """Perfect-recall sequential individual follower (see module docstring)."""
    predictor.model.eval()
    device = noises.x2.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    grid1 = _symmetric_grid(bound, cfg.grid_size_x1, device)
    grid4 = _symmetric_grid(bound, cfg.grid_size_x4, device)

    # Stage 1: each individual observes (x2, eps1) and chooses delta_1.
    if scm.intervene_on_x1:
        delta1, _ = _stage1_best(
            scm,
            predictor,
            objective,
            x2=noises.x2,
            eps1=noises.eps1,
            grid1=grid1,
            grid4=grid4,
            cfg=cfg,
            generator=generator,
        )
    else:
        delta1 = torch.zeros_like(noises.x2)
    x1, y, x3 = _stage1_forward(
        scm, noises.x2, noises.eps1, noises.eps_y, noises.eps3, delta1
    )

    # Stage 2: each individual observes (x1, x2, y, x3, eps4) and chooses delta_4.
    if uses_x4_or_x5(predictor):
        eps5_draws = torch.randn(
            noises.n, cfg.mc_stage2, 1, device=device, generator=generator
        )
        delta4, _ = _stage2_best(
            scm,
            predictor,
            objective,
            x1=x1,
            x2=noises.x2,
            y=y,
            x3=x3,
            eps4=noises.eps4,
            grid4=grid4,
            eps5_draws=eps5_draws,
            max_rows=cfg.predict_max_rows,
        )
    else:
        delta4 = torch.zeros_like(noises.x2)
    x4, x5 = _stage2_forward(scm, y, noises.x2, noises.eps4, noises.eps5, delta4)
    x = _assemble_x(x1, noises.x2, x3, x4, x5)
    return _outcome(name, predictor, objective, x, y, delta1, delta4)


@torch.no_grad()
def omniscient_individual_follower(
    scm: SCM,
    predictor: PredictorBundle,
    objective: AttackObjective,
    noises: Noises,
    bound: float,
    cfg: IndividualFollowerConfig,
    name: str = "individual_omniscient",
) -> FollowerOutcome:
    """Individual who knows all of their own noises; maximizes the realized payoff."""
    predictor.model.eval()
    device = noises.x2.device
    n = noises.n
    size = cfg.omniscient_grid_size
    grid1 = (
        _symmetric_grid(bound, size, device)
        if scm.intervene_on_x1
        else torch.zeros(1, device=device)
    )
    grid4 = (
        _symmetric_grid(bound, size, device)
        if uses_x4_or_x5(predictor)
        else torch.zeros(1, device=device)
    )
    g1, g4 = grid1.numel(), grid4.numel()
    delta1_star = torch.zeros(n, 1, device=device)
    delta4_star = torch.zeros(n, 1, device=device)
    chunk = max(1, cfg.predict_max_rows // (g1 * g4))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        n_c = stop - start
        x2c, eps1c, eps_yc, eps3c, eps4c, eps5c = (
            t[start:stop].view(n_c, 1, 1, 1)
            for t in (
                noises.x2,
                noises.eps1,
                noises.eps_y,
                noises.eps3,
                noises.eps4,
                noises.eps5,
            )
        )
        d1 = grid1.view(1, g1, 1, 1)
        d4 = grid4.view(1, 1, g4, 1)
        x1, y, x3 = _stage1_forward(scm, x2c, eps1c, eps_yc, eps3c, d1)
        x4, x5 = _stage2_forward(scm, y, x2c, eps4c, eps5c, d4)
        x_full = _assemble_x(x1, x2c, x3, x4, x5)  # (n_c, G1, G4, 5)
        pred = _predict(predictor, x_full, cfg.predict_max_rows)
        payoff = follower_payoff(y, pred, objective).view(n_c, g1 * g4)
        idx = payoff.argmax(dim=1)
        delta1_star[start:stop] = grid1[idx // g4].view(n_c, 1)
        delta4_star[start:stop] = grid4[idx % g4].view(n_c, 1)
    x1, y, x3 = _stage1_forward(
        scm, noises.x2, noises.eps1, noises.eps_y, noises.eps3, delta1_star
    )
    x4, x5 = _stage2_forward(scm, y, noises.x2, noises.eps4, noises.eps5, delta4_star)
    x = _assemble_x(x1, noises.x2, x3, x4, x5)
    return _outcome(name, predictor, objective, x, y, delta1_star, delta4_star)
