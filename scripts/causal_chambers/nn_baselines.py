"""Neural-network baselines (ERM / IRM / GroupDRO, linear or deep) for the
causal-chambers prediction-intervention experiment.

Adapted from the user's reference scripts (``irm_nn.py`` / ``irm_linear.py``).  A single
training loop handles all three methods; only the loss differs, so each (method,
architecture) pair shares the exact same network.  Exposes the sklearn-style
``fit(X, y, E)`` / ``predict_proba(X)`` interface used by the harness for ``f_sc`` /
``f_csc``.

Design choices (see the project plan):
  * **IRM** uses the bounded convex combination  ``L = (1-lambda)*ERM + lambda*penalty``
    with ``lambda in [0, 1]`` for BOTH linear and deep (the InvarianceUnitTests form).
    Because it is bounded, no colored-MNIST loss rescaling is needed.
  * Train on **BCE**, but **select hyperparameters by worst-environment Brier** (the
    deployment / leader metric), consistent with how ``f_csc`` is selected.
  * **Retrain to the epoch where validation was actually best** (not best + patience).
  * **Seed-ensemble**: retrain the chosen config with ``n_seeds`` seeds and average
    ``predict_proba`` -> one stable curve per method.
  * Deep MLP has **no BatchNorm** (only 7 features).  Full-batch Adam.

ERM / IRM / GroupDRO all use every feature and the environment labels (ERM ignores E).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# network
# ──────────────────────────────────────────────────────────────────────────────


class TabularMLP(nn.Module):
    """Linear model or small MLP for low-dim tabular binary classification.

    ``architecture="linear"`` -> a single ``nn.Linear(n_features, 1)``.
    ``architecture="deep"``   -> ``depth`` hidden layers (ReLU, no BatchNorm) + head.
    """

    def __init__(
        self,
        n_features: int,
        architecture: str = "deep",
        hidden_width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if architecture == "linear":
            self.net: nn.Module = nn.Linear(n_features, 1)
        elif architecture == "deep":
            layers: list[nn.Module] = []
            in_dim = n_features
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, hidden_width))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_width
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown architecture: {architecture!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """IRMv1 penalty: squared gradient of the BCE w.r.t. a dummy unit scale.

    Ported from ``irm_nn.py``.
    """
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits * scale, y)
    (g,) = torch.autograd.grad(loss, scale, create_graph=True)
    return g.pow(2).sum()


# ──────────────────────────────────────────────────────────────────────────────
# hyperparameters
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Hparams:
    method: str = "erm"  # "erm" | "irm" | "groupdro"
    architecture: str = "deep"  # "linear" | "deep"
    hidden_width: int = 64
    depth: int = 2
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    irm_lambda: float = 0.5  # convex weight in [0, 1]
    groupdro_eta: float = 0.01
    anneal_iters: int = 0  # ERM-warmup epochs before the IRM penalty turns on
    n_epochs: int = 500
    patience: int = 100
    seed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# helpers (from the reference scripts)
# ──────────────────────────────────────────────────────────────────────────────


def _standardise(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-scoring fitted on X_train."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, mu, sigma


def _split_by_env(
    X: np.ndarray, y: np.ndarray, E: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    env_ids = np.sort(np.unique(E))
    return [X[E == e] for e in env_ids], [y[E == e] for e in env_ids]


def _split_train_val(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    val_frac: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """80/20 split within each environment."""
    rng = np.random.RandomState(seed)
    Xtr, ytr, Xva, yva = [], [], [], []
    for X_e, y_e in zip(X_envs, y_envs, strict=True):
        idx = rng.permutation(len(X_e))
        cut = int(len(X_e) * (1 - val_frac))
        Xtr.append(X_e[idx[:cut]])
        ytr.append(y_e[idx[:cut]])
        Xva.append(X_e[idx[cut:]])
        yva.append(y_e[idx[cut:]])
    return Xtr, ytr, Xva, yva


def _brier(y_true: np.ndarray, p1: np.ndarray) -> float:
    return float(np.mean((p1 - y_true) ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# training (one loop, three losses)
# ──────────────────────────────────────────────────────────────────────────────


def _train(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    hp: _Hparams,
    X_val_envs: list[np.ndarray] | None = None,
    y_val_envs: list[np.ndarray] | None = None,
    device: str = "cpu",
) -> tuple[TabularMLP, int]:
    """Train one model.  Returns (model, best_epoch).

    Loss by method (per full-batch step):
      * erm:      pooled BCE over all samples.
      * irm:      (1-lambda)*mean_e BCE_e + lambda*mean_e penalty_e  (lambda in [0,1]);
                  pure-ERM warmup for the first ``anneal_iters`` epochs.
      * groupdro: online group weights  q_e <- q_e * exp(eta * BCE_e);  loss = sum_e q_e BCE_e.

    With a validation set, early-stops on **worst-environment Brier** and restores the
    best epoch's weights (tracking the epoch at which it was achieved).
    """
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    n_features = X_envs[0].shape[1]
    model = TabularMLP(
        n_features, hp.architecture, hp.hidden_width, hp.depth, hp.dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    Xt = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_envs]
    yt = [torch.tensor(y, dtype=torch.float32, device=device) for y in y_envs]
    all_x = torch.cat(Xt, dim=0)
    all_y = torch.cat(yt, dim=0)

    has_val = X_val_envs is not None and y_val_envs is not None
    if has_val:
        assert X_val_envs is not None and y_val_envs is not None
        Xv = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_val_envs]
        yv_np = [np.asarray(y) for y in y_val_envs]
    else:
        Xv, yv_np = [], []

    q = torch.ones(len(Xt), device=device) / len(Xt)  # GroupDRO weights
    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = hp.n_epochs
    patience_counter = 0

    for epoch in range(hp.n_epochs):
        model.train()

        if hp.method == "erm":
            loss = F.binary_cross_entropy_with_logits(model(all_x).squeeze(-1), all_y)
        else:
            env_losses = []
            env_pens = []
            for x_e, y_e in zip(Xt, yt, strict=True):
                logits_e = model(x_e).squeeze(-1)
                env_losses.append(F.binary_cross_entropy_with_logits(logits_e, y_e))
                if hp.method == "irm":
                    env_pens.append(irm_penalty(logits_e, y_e))

            if hp.method == "irm":
                erm = torch.stack(env_losses).mean()
                if epoch < hp.anneal_iters:
                    loss = erm  # warmup: pure ERM
                else:
                    pen = torch.stack(env_pens).mean()
                    loss = (1.0 - hp.irm_lambda) * erm + hp.irm_lambda * pen
            elif hp.method == "groupdro":
                losses = torch.stack(env_losses)
                with torch.no_grad():
                    q = q * torch.exp(hp.groupdro_eta * losses.detach())
                    q = q / q.sum()
                loss = (q * losses).sum()
            else:
                raise ValueError(f"Unknown method: {hp.method!r}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        # early stopping on worst-env val Brier, only after the IRM warmup
        if has_val and (epoch + 1) % 10 == 0 and (epoch + 1) > hp.anneal_iters:
            model.eval()
            with torch.no_grad():
                worst = 0.0
                for x_v, y_v in zip(Xv, yv_np, strict=True):
                    p1 = torch.sigmoid(model(x_v).squeeze(-1)).cpu().numpy()
                    worst = max(worst, _brier(y_v, p1))
            if worst < best_metric - 1e-6:
                best_metric = worst
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 10
                if patience_counter >= hp.patience:
                    break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def _predict_proba_nn(
    model: TabularMLP, X: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    return torch.sigmoid(model(Xt).squeeze(-1)).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# grid search (selection on worst-environment Brier)
# ──────────────────────────────────────────────────────────────────────────────


def _build_grid(method: str, architecture: str) -> list[dict]:
    grid: dict[str, list] = {"lr": [1e-3, 5e-4], "weight_decay": [0.0, 1e-4]}
    if architecture == "deep":
        grid["depth"] = [2, 3]
    if method == "irm":
        grid["irm_lambda"] = [0.1, 0.5, 0.9, 0.99]
    elif method == "groupdro":
        grid["groupdro_eta"] = [1e-2, 1e-1]
    keys = list(grid)
    return [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*grid.values())
    ]


def _make_hp(cfg: dict, base: "_Hparams", n_epochs: int, seed: int) -> _Hparams:
    return _Hparams(
        method=base.method,
        architecture=base.architecture,
        hidden_width=base.hidden_width,
        depth=cfg.get("depth", base.depth),
        dropout=base.dropout,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        irm_lambda=cfg.get("irm_lambda", base.irm_lambda),
        groupdro_eta=cfg.get("groupdro_eta", base.groupdro_eta),
        anneal_iters=base.anneal_iters if base.method == "irm" else 0,
        n_epochs=n_epochs,
        patience=base.patience,
        seed=seed,
    )


def _evaluate_single_config(
    cfg: dict,
    base: _Hparams,
    X_tr: list[np.ndarray],
    y_tr: list[np.ndarray],
    X_va: list[np.ndarray],
    y_va: list[np.ndarray],
    device: str,
) -> tuple[dict, float, int]:
    hp = _make_hp(cfg, base, n_epochs=base.n_epochs, seed=base.seed)
    model, best_epoch = _train(X_tr, y_tr, hp, X_va, y_va, device=device)
    worst = 0.0
    for x_v, y_v in zip(X_va, y_va, strict=True):
        p1 = _predict_proba_nn(model, x_v, device)
        worst = max(worst, _brier(np.asarray(y_v), p1))
    return cfg, float(worst), int(best_epoch)


# ──────────────────────────────────────────────────────────────────────────────
# sklearn-style classifier
# ──────────────────────────────────────────────────────────────────────────────


class NNBaselineClassifier(ClassifierMixin, BaseEstimator):
    """ERM / IRM / GroupDRO neural-network baseline (linear or deep).

    Grid-searches hyperparameters (selection: worst-environment validation Brier),
    then retrains the chosen config on all data for the validation-optimal number of
    epochs with ``n_seeds`` seeds, averaging ``predict_proba`` across them.

    Parameters
    ----------
    method : {"erm", "irm", "groupdro"}
    architecture : {"linear", "deep"}
    n_seeds : int, default=3
        Seeds to ensemble in the final refit.
    """

    def __init__(
        self,
        method: str = "erm",
        architecture: str = "deep",
        n_seeds: int = 3,
        hidden_width: int = 64,
        anneal_iters: int = 100,
        n_epochs: int = 500,
        patience: int = 100,
        val_frac: float = 0.2,
        n_jobs: int = 1,
        device: str = "cpu",
        verbose: bool = False,
        random_state: int = 123,
    ):
        self.method = method
        self.architecture = architecture
        self.n_seeds = n_seeds
        self.hidden_width = hidden_width
        self.anneal_iters = anneal_iters
        self.n_epochs = n_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.n_jobs = n_jobs
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def _base_hp(self) -> _Hparams:
        return _Hparams(
            method=self.method,
            architecture=self.architecture,
            hidden_width=self.hidden_width,
            anneal_iters=self.anneal_iters,
            n_epochs=self.n_epochs,
            patience=self.patience,
            seed=self.random_state,
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, environment: np.ndarray
    ) -> "NNBaselineClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        environment = np.asarray(environment)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        X_s, mu, sigma = _standardise(X)
        self._mu, self._sigma = mu, sigma
        X_envs, y_envs = _split_by_env(X_s, y, environment)

        base = self._base_hp()
        grid = _build_grid(self.method, self.architecture)
        X_tr, y_tr, X_va, y_va = _split_train_val(
            X_envs, y_envs, self.val_frac, seed=self.random_state
        )
        if self.verbose:
            print(
                f"  Fitting {self.method}-{self.architecture}: "
                f"{len(grid)} configs, n_jobs={self.n_jobs}"
            )

        raw = Parallel(n_jobs=self.n_jobs)(
            delayed(_evaluate_single_config)(
                cfg, base, X_tr, y_tr, X_va, y_va, self.device
            )
            for cfg in grid
        )
        results = [r for r in raw if r is not None]
        self._grid_results = results

        best_cfg, best_worst, best_epoch = min(results, key=lambda r: r[1])
        self.best_config_ = best_cfg
        self.best_n_epochs_ = best_epoch
        self.best_worst_val_brier_ = best_worst
        if self.verbose:
            print(
                f"    best worst-env val Brier={best_worst:.4f}, "
                f"epochs={best_epoch}, cfg={best_cfg}"
            )

        # seed-ensemble: retrain on ALL data for best_epoch epochs (no val -> no early stop)
        self._models = []
        for s in range(self.n_seeds):
            hp = _make_hp(
                best_cfg, base, n_epochs=best_epoch, seed=self.random_state + s
            )
            model, _ = _train(X_envs, y_envs, hp, device=self.device)
            self._models.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_s = (X - self._mu) / self._sigma
        p1 = np.mean(
            [_predict_proba_nn(m, X_s, self.device) for m in self._models], axis=0
        )
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
