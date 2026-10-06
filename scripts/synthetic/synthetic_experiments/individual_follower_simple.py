"""Compact individual followers (Section 2.1 of the paper), self-contained.

Each individual is one draw of the exogenous noise.  The follower acts in two
steps, because X1 -> Y -> X4:

    step 1: observe (x2, eps1)               -> choose delta_1 in [-b, b]
    step 2: observe (x1, x2, y, x3, eps4)    -> choose delta_4 in [-b, b]

Both choices are made by grid search; expectations over noise that is not yet
realized are Monte Carlo averages.  Options:

    greedy=True        choose delta_1 as if no intervention on X4 followed;
    greedy=False       anticipate the optimal delta_4 (backward induction; exact);
    observe_x3=False   step 2 does not observe x3 (integrated out);
    one_shot_x2=True   literal Section-2.1 follower with O = {X2}: (delta_1, delta_4)
                       are chosen jointly after observing (x2, eps1, eps4), so
                       delta_4 cannot react to Y.
    observe_x1=False   step 2 does not observe x1: X1 is integrated out under its
                       posterior given (y, x2) by importance sampling (only
                       meaningful without an intervention on X1).

``objective`` is MSE (maximize (Y - f)^2) or PREDICTION (minimize f).
"""

from __future__ import annotations

import torch
from torch import Tensor

MSE, PREDICTION = 0, 1


def payoff(y: Tensor, pred: Tensor, objective: int) -> Tensor:
    """Per-sample payoff the individual maximizes."""
    return (y - pred) ** 2 if objective == MSE else -pred


def predict(predictor, x: Tensor, max_rows: int = 1 << 18) -> Tensor:
    flat = x.reshape(-1, x.shape[-1])
    out = [
        predictor.model(predictor.select_and_standardize(c))
        for c in flat.split(max_rows)
    ]
    return torch.cat(out).reshape(*x.shape[:-1], 1)


def _cat(*cols: Tensor) -> Tensor:
    shape = torch.broadcast_shapes(*(c.shape for c in cols))
    return torch.cat([c.expand(shape) for c in cols], dim=-1)


def forward(scm, x2, eps1, eps_y, eps3, eps4, eps5, delta1, delta4):
    """SCM forward pass with additive shifts; all arguments broadcast."""
    x1 = scm._clean_x1(x2, eps1) + delta1
    y = scm._clean_y(x1, x2, eps_y)
    x3 = scm._clean_x3(y, eps3)
    x4 = scm._clean_x4_from_parents(y, x2, eps4) + delta4
    x5 = scm._clean_x5(x4, eps5)
    x = _cat(x1, x2, x3, x4, x5)
    return x, y.expand(*x.shape[:-1], 1)


def _flat(t: Tensor, shape) -> Tensor:
    return t.expand(shape).reshape(-1, 1)


def noises_tuple(noises):
    return (noises.x2, noises.eps1, noises.eps_y, noises.eps3, noises.eps4, noises.eps5)


@torch.no_grad()
def step2(scm, predictor, objective, x1, x2, y, x3, eps4, grid, n_mc, gen):
    """Optimal delta_4 and its value for states (B, 1); grid (G,); MC over eps5.

    If ``x3`` is None, X3 is not observed and is integrated out as well.  If
    ``x1`` is None, X1 is not observed either: it is integrated out under its
    posterior given (y, x2) with self-normalized importance weights (requires
    that X1 is not intervened on, so that x1 = clean_x1(x2, eps1)).
    """
    b = x2.shape[0]
    v = lambda t: t.view(b, 1, 1, 1)  # noqa: E731
    eps5 = torch.randn(b, 1, n_mc, 1, generator=gen)
    x4 = scm._clean_x4_from_parents(v(y), v(x2), v(eps4)) + grid.view(1, -1, 1, 1)
    x5 = scm._clean_x5(x4, eps5)
    if x3 is None:
        x3 = scm._clean_x3(v(y), torch.randn(b, 1, n_mc, 1, generator=gen))
    else:
        x3 = v(x3)
    if x1 is None:
        x1 = scm._clean_x1(v(x2), torch.randn(b, 1, n_mc, 1, generator=gen))
        zero = torch.zeros_like(x1)
        mean_y = scm._clean_y(x1, v(x2), zero)
        sd_y = scm._clean_y(x1, v(x2), zero + 1.0) - mean_y  # additive noise scale
        log_w = -0.5 * ((v(y) - mean_y) / sd_y) ** 2
        weights = torch.softmax(log_w, dim=2)  # (B, 1, M, 1), sums to 1 over draws
    else:
        x1 = v(x1)
        weights = torch.full((1, 1, n_mc, 1), 1.0 / n_mc)
    x = _cat(x1, v(x2), x3, x4, x5)
    value = (
        (payoff(v(y), predict(predictor, x), objective) * weights).sum(2).squeeze(-1)
    )
    best, idx = value.max(1)
    return grid[idx].view(b, 1), best.view(b, 1)


@torch.no_grad()
def step1(
    scm,
    predictor,
    objective,
    x2,
    eps1,
    grid,
    n_mc,
    gen,
    greedy,
    observe_x3=True,
    n_mc2=4,
    chunk=256,
):
    """Optimal delta_1 for individuals (n, 1); MC over (eps_y, eps3, eps4, eps5)."""
    out = torch.zeros_like(x2)
    g = grid.view(1, -1, 1, 1)
    for s in range(0, x2.shape[0], chunk):
        x2c, eps1c = (t[s : s + chunk].view(-1, 1, 1, 1) for t in (x2, eps1))
        n = x2c.shape[0]
        eps_y, eps3, eps4, eps5 = (
            torch.randn(n, 1, n_mc, 1, generator=gen) for _ in range(4)
        )
        if greedy:  # value of delta_1 assuming delta_4 = 0
            x, y = forward(scm, x2c, eps1c, eps_y, eps3, eps4, eps5, g, 0.0)
            value = payoff(y, predict(predictor, x), objective).mean(2).squeeze(-1)
        else:  # value of delta_1 anticipating the optimal delta_4 (backward induction)
            x1 = scm._clean_x1(x2c, eps1c) + g
            y = scm._clean_y(x1, x2c, eps_y)
            shape = (n, grid.numel(), n_mc, 1)
            x3 = _flat(scm._clean_x3(y, eps3), shape) if observe_x3 else None
            _, value = step2(
                scm,
                predictor,
                objective,
                _flat(x1, shape),
                _flat(x2c, shape),
                _flat(y, shape),
                x3,
                _flat(eps4, shape),
                grid,
                n_mc2,
                gen,
            )
            value = value.view(n, grid.numel(), n_mc).mean(2)
        out[s : s + chunk] = grid[value.argmax(1)].view(n, 1)
    return out


@torch.no_grad()
def one_shot(scm, predictor, objective, x2, eps1, eps4, grid, n_mc, gen, chunk=64):
    """(delta_1, delta_4) chosen jointly after observing (x2, eps1, eps4)."""
    d1_out, d4_out = torch.zeros_like(x2), torch.zeros_like(x2)
    g1 = grid.view(1, -1, 1, 1, 1)
    g4 = grid.view(1, 1, -1, 1, 1)
    for s in range(0, x2.shape[0], chunk):
        x2c, eps1c, eps4c = (
            t[s : s + chunk].view(-1, 1, 1, 1, 1) for t in (x2, eps1, eps4)
        )
        n = x2c.shape[0]
        eps_y, eps3, eps5 = (
            torch.randn(n, 1, 1, n_mc, 1, generator=gen) for _ in range(3)
        )
        x, y = forward(scm, x2c, eps1c, eps_y, eps3, eps4c, eps5, g1, g4)
        value = payoff(y, predict(predictor, x), objective).mean(3).view(n, -1)
        idx = value.argmax(1)
        d1_out[s : s + chunk] = grid[idx // grid.numel()].view(n, 1)
        d4_out[s : s + chunk] = grid[idx % grid.numel()].view(n, 1)
    return d1_out, d4_out


@torch.no_grad()
def individual_follower(
    scm,
    predictor,
    objective,
    noises,
    bound,
    *,
    grid_size=13,
    n_mc=16,
    n_mc_step2=4,
    greedy=False,
    observe_x3=True,
    observe_x1=True,
    one_shot_x2=False,
    n_mc_final=None,
    seed=0,
):
    """Run the individual follower on fixed noises; returns (x, y, delta1, delta4).

    ``n_mc`` Monte Carlo draws are used for step 1 (over eps_Y, eps3, eps4, eps5)
    and ``n_mc_step2`` for the step-2 evaluations inside step 1.  The final
    step-2 choice uses ``n_mc_final`` draws (default: ``n_mc_step2``); it is a
    single pass, so a large value costs little.
    """
    gen = torch.Generator().manual_seed(seed)
    grid = torch.linspace(-bound, bound, grid_size)
    uses_x4 = bool(set(predictor.subset_indices) & {3, 4})
    zero = torch.zeros_like(noises.x2)
    if one_shot_x2 and uses_x4 and scm.intervene_on_x1:
        d1, d4 = one_shot(
            scm,
            predictor,
            objective,
            noises.x2,
            noises.eps1,
            noises.eps4,
            grid,
            n_mc,
            gen,
        )
    else:
        if scm.intervene_on_x1:
            d1 = step1(
                scm,
                predictor,
                objective,
                noises.x2,
                noises.eps1,
                grid,
                n_mc,
                gen,
                greedy=greedy or not uses_x4,
                observe_x3=observe_x3,
                n_mc2=n_mc_step2,
            )
        else:
            d1 = zero
        x1 = scm._clean_x1(noises.x2, noises.eps1) + d1
        y = scm._clean_y(x1, noises.x2, noises.eps_y)
        x3 = scm._clean_x3(y, noises.eps3) if observe_x3 else None
        if not observe_x1 and scm.intervene_on_x1:
            raise ValueError("observe_x1=False requires intervene_on_x1=False.")
        if uses_x4:
            d4, _ = step2(
                scm,
                predictor,
                objective,
                x1 if observe_x1 else None,
                noises.x2,
                y,
                x3,
                noises.eps4,
                grid,
                n_mc_final if n_mc_final is not None else n_mc_step2,
                gen,
            )
        else:
            d4 = zero  # the payoff does not depend on delta_4
    x, y = forward(scm, *noises_tuple(noises), d1, d4)
    return x, y, d1, d4
