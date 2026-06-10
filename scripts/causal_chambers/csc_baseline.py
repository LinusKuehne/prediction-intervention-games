"""Causal Strategic Classification (CSERM, "Learning with clean data") baseline.

Faithful reimplementation of the clean-data method (Algorithm 1 / Eq. 16) from

    Horowitz & Rosenfeld, "Causal Strategic Classification: A Tale of Two Shifts",
    ICML 2023  (code: CSC/, https://github.com/guyhorowitz/CSC).

The formulas are ported verbatim from the authors' code (see the inline ``CSC/...``
references):
  * ``CSC/delta.py``    : ``QuadraticCost`` / ``QuadraticCostDelta`` (closed-form best
                          response for a linear ``f`` + quadratic cost) and
                          ``create_cost_functions``.
  * ``CSC/model.py``    : ``s_hinge_loss`` (strategic hinge) and ``hinge_loss``.
  * ``CSC/training.py`` : ``CausalStrategicTrainer`` soft-label rule
                          (``get_h_scores`` / ``get_labels_for_hinge``) and the
                          warm-start + strategic training loop.
  * ``CSC/data.py``     : feature standardization (center, /std, /sqrt(d)).

Adaptations for the Causal Chambers prediction-intervention game (all documented in
the project plan):

1. Only ``ir_3`` and ``vis_3`` are manipulable, so we use a *masked* diagonal cost:
   ``Q^{-1} = diag(mask) / alpha`` with ``mask = 1`` on the movable coordinates and
   ``0`` elsewhere (infinite cost otherwise).  The closed-form best response then moves
   only those coordinates, and the strategic-hinge max-gain term uses
   ``||w_movable|| = sqrt(2 * w^T Q^{-1} w)`` instead of the full ``||w||``.

2. The chambers follower MINIMIZES ``E[f]`` whereas CSC users MAXIMIZE the prediction.
   We relabel ``s = 1 - 2*Y`` (so ``s = +1`` <=> ``Y = 0`` is "favorable"), train ``f``
   to predict ``s``, and report ``P(Y=1) = 1 - sigma(score)`` via a simple 1-D Platt
   calibration so the Brier score is comparable to the RF / SC predictors.

3. ``h`` is a RandomForest (CSC allows any function class for ``h``).  Because the
   causal coordinates ``x_c = RGB`` never move (only ``ir_3, vis_3`` do), the causal
   label-update component of CSERM is largely inactive: ``h(x_c^f, x_r) = h(x_c, x_r)``,
   so the soft labels stay ~ the clean labels.

4. ``alpha`` (cost scale) is calibrated so that ~50% of *unfavorable* training points
   can flip to favorable under the initial ERM classifier (Horowitz-Rosenfeld), then
   selected over the grid by *worst-environment* validation Brier across the training
   environments only (never the 49 deployment actions).

The estimator exposes the sklearn ``fit(X, y, E)`` / ``predict_proba(X)`` interface used
by the chambers harness (``scripts/causal_chambers/adversarial_follower_dspur.py``), so
it is evaluated by the exact same follower + metrics as every other predictor.
"""

import math

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.optim import Adam


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class _QuadraticCostDelta:
    """Closed-form best response for a linear ``f`` with a masked quadratic cost.

    Ported from ``CSC/delta.py`` (``QuadraticCostDelta``, lines 55-99) with
    ``Q^{-1} = diag(mask) / alpha`` so that only the movable coordinates move.  For an
    unfavorable point the move sends it exactly to the decision boundary; the movement
    indicator is ``1{0 < cost <= 2}`` (exact) or its sigmoid relaxation (approx).
    """

    def __init__(self, cls: nn.Linear, cost_matrix, cost_matrix_inv, tau):
        self.cls = cls
        self.Q = cost_matrix
        self.Qinv = cost_matrix_inv
        self.tau = tau

    def _step_size(self, x):
        w = self.cls.weight  # (1, d)
        denom = w @ (self.Qinv @ w.T) + 1e-8  # (1, 1); CSC delta.py:61
        step = self.cls(x) / denom  # (N, 1)
        return torch.minimum(torch.zeros_like(step), step)  # CSC delta.py:63

    def _x_tag_and_cost(self, x):
        step = self._step_size(x)
        direction = (self.Qinv @ self.cls.weight.T).T  # (1, d); CSC delta.py:82
        x_tag = x - step @ direction  # (N, d)
        diff = x - x_tag
        cost = ((diff @ self.Q) * diff).sum(
            dim=1, keepdim=True
        )  # (N, 1); CSC delta.py:48-52
        return x_tag, cost

    def approx_ind_for_h(self, x):
        # CSC delta.py:75-78 (__approx_ind_for_h), returned by delta.approx(x, True).
        _, cost = self._x_tag_and_cost(x)
        below_2 = torch.sigmoid((2 - cost) * self.tau)
        above_0 = (torch.sigmoid(cost * self.tau) - 0.5) * 2
        return below_2 * above_0

    def exact_ind(self, x):
        # CSC delta.py:66-69 (__exact_ind).
        _, cost = self._x_tag_and_cost(x)
        return (cost <= 2).float() * (cost > 0).float()


class CausalStrategicClassifier(BaseEstimator, ClassifierMixin):
    """Clean-data CSERM strategic classifier with a restricted (masked) cost.

    Parameters
    ----------
    features : sequence of str
        Column names of ``X`` in the order they are passed to ``fit``/``predict_proba``
        (must match the order produced by the harness' ``build_X``).
    causal_features : sequence of str
        The causal subset ``x_c`` (immovable here); used only for documentation and a
        disjointness sanity check against ``movable_features``.
    movable_features : sequence of str
        Coordinates the follower can manipulate (finite cost); everything else has
        infinite cost.  For the chambers game this is ``("ir_3", "vis_3")``.
    alpha_mults : sequence of float
        Grid of multipliers applied to the calibrated ``alpha_0``.
    tau, epochs, lr, batch_size, f_reg
        Training hyper-parameters (CSC defaults: tau=4, lr=0.01, epochs=100, bs=64).
    n_h_estimators : int
        Number of trees for the RandomForest ``h``.
    val_frac : float
        Per-environment fraction held out for alpha selection.
    random_state : int
        Seed for the torch model init, mini-batch order, RF, and the split.
    """

    def __init__(
        self,
        features,
        causal_features=("red", "green", "blue"),
        movable_features=("ir_3", "vis_3"),
        alpha_mults=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        tau=4.0,
        epochs=100,
        lr=0.01,
        batch_size=64,
        f_reg=0.0,
        n_h_estimators=500,
        val_frac=0.3,
        random_state=123,
    ):
        self.features = list(features)
        self.causal_features = list(causal_features)
        self.movable_features = list(movable_features)
        self.alpha_mults = tuple(alpha_mults)
        self.tau = tau
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.f_reg = f_reg
        self.n_h_estimators = n_h_estimators
        self.val_frac = val_frac
        self.random_state = random_state

    # ---- standardization (CSC/data.py:28-30) -------------------------------------

    def _standardize_fit(self, X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        d = X.shape[1]
        self._mu, self._sigma, self._d = mu, sigma, d
        return ((X - mu) / sigma) / math.sqrt(d)

    def _standardize(self, X):
        return ((X - self._mu) / self._sigma) / math.sqrt(self._d)

    # ---- losses ------------------------------------------------------------------

    def _s_hinge(self, X, y, f, alpha, mask_t):
        # CSC/model.py:40-43, with ||W|| -> ||w_movable|| = sqrt(2 * w^T Q^{-1} w).
        w = f.weight[0]
        w_mov_norm = torch.norm(w * mask_t, p=2)
        max_cost = math.sqrt(2.0 / alpha)
        return torch.mean(
            torch.relu(
                1 - y.view(-1) * f(X).view(-1) - max_cost * y.view(-1) * w_mov_norm
            )
        )

    # ---- training ----------------------------------------------------------------

    def _run_epochs(
        self, f, opt, Xt, st, hlt, delta, alpha, mask_t, strategic, patience=7
    ):
        n = Xt.shape[0]
        gen = torch.Generator()
        gen.manual_seed(self.random_state)
        best, bad = float("inf"), 0
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            tot, nb = 0.0, 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, sb = Xt[idx], st[idx]
                opt.zero_grad()
                if strategic:
                    move_ind = delta.approx_ind_for_h(
                        xb
                    )  # CSC: delta.approx(Xbatch, True)
                    soft = sb + (hlt[idx] - sb) * move_ind  # CSC: get_labels_for_hinge
                    loss = self._s_hinge(xb, soft, f, alpha, mask_t)
                else:
                    # warm start: plain hinge on hard labels (CSC hinge_loss, model.py:9-13)
                    loss = torch.mean(torch.relu(1 - sb.view(-1) * f(xb).view(-1)))
                if self.f_reg:
                    loss = loss + self.f_reg * torch.norm(f.weight)
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            avg = tot / max(nb, 1)
            if avg < best - 1e-5:
                best, bad = avg, 0
            else:
                bad += 1
                if bad >= patience:
                    break

    def _fit_linear(self, X_std, s, alpha, h_labels, strategic):
        d = X_std.shape[1]
        _set_seed(self.random_state)
        f = nn.Linear(d, 1)
        opt = Adam(f.parameters(), lr=self.lr)  # single optimizer for both phases (CSC)
        st = torch.tensor(s, dtype=torch.float32).view(-1, 1)
        mask_t = torch.tensor(self.movable_mask_, dtype=torch.float32)
        mask_bool = self.movable_mask_.astype(bool)

        if strategic and alpha == 0.0:
            # alpha -> 0 limit: the strategic-hinge penalty sqrt(2/alpha)*||w_mov|| blows
            # up unless ||w_movable|| = 0, i.e. the movable features (ir_3, vis_3) are
            # dropped entirely.  Fit a plain-hinge linear model on the non-movable columns
            # and force the movable weights to exactly 0 (so predict ignores ir_3, vis_3).
            X_masked = X_std.copy()
            X_masked[:, mask_bool] = 0.0
            Xt0 = torch.tensor(X_masked, dtype=torch.float32)
            self._run_epochs(
                f, opt, Xt0, st, None, None, alpha, mask_t, strategic=False
            )
            with torch.no_grad():
                f.weight[0, torch.from_numpy(mask_bool)] = 0.0
            return f

        Xt = torch.tensor(X_std, dtype=torch.float32)
        # warm-start: non-strategic training (CSC: init_f_with_non_strategic_training)
        self._run_epochs(f, opt, Xt, st, None, None, alpha, mask_t, strategic=False)
        if strategic:
            Q = torch.eye(d) * alpha  # CSC create_cost_functions, cost matrix
            Qinv = torch.diag(
                mask_t / alpha
            )  # masked inverse cost (only movable coords)
            delta = _QuadraticCostDelta(f, Q, Qinv, self.tau)
            hlt = torch.tensor(h_labels, dtype=torch.float32).view(-1, 1)
            self._run_epochs(f, opt, Xt, st, hlt, delta, alpha, mask_t, strategic=True)
        return f

    # ---- h (RandomForest), alpha calibration, Platt, selection -------------------

    def _fit_h_labels(self, X_std, y):
        """Train RF ``h`` and return soft labels ``2*P(favorable) - 1`` in [-1, 1].

        Evaluated at the clean point: the causal coords never move, so ``h(x_c^f, x_r)``
        equals ``h`` on the original full feature vector (CSC get_h_scores).
        """
        fav = (y == 0).astype(int)  # favorable = Y=0 (matches s=+1)
        rf = RandomForestClassifier(
            n_estimators=self.n_h_estimators, random_state=self.random_state
        )
        rf.fit(X_std, fav)
        if len(rf.classes_) < 2:
            p_fav = np.full(len(y), float(rf.classes_[0]))
        else:
            p_fav = rf.predict_proba(X_std)[:, list(rf.classes_).index(1)]
        return 2.0 * p_fav - 1.0

    def _calibrate_alpha0(self, X_fit_std, s_fit):
        """alpha_0 = median over unfavorable points of 2*||w0_mov||^2 / score^2.

        With the fixed cost budget of 2 (CSC), a point with score < 0 can flip iff
        alpha <= 2*||w0_mov||^2 / score^2, so the median gives ~50% movable.
        """
        f0 = self._fit_linear(
            X_fit_std, s_fit, alpha=1.0, h_labels=None, strategic=False
        )
        w0 = f0.weight.detach().numpy().ravel()
        b0 = float(f0.bias.detach().item())
        scores = X_fit_std @ w0 + b0
        w_mov_sq = float(np.sum((w0 * self.movable_mask_) ** 2))
        unfav = scores < 0
        if not unfav.any() or w_mov_sq == 0.0:
            return 1.0
        alpha_i = 2.0 * w_mov_sq / (scores[unfav] ** 2)
        return float(np.median(alpha_i))

    @staticmethod
    def _fit_platt(scores, y):
        return LogisticRegression().fit(scores.reshape(-1, 1), y)

    @staticmethod
    def _p1(scores, platt):
        return platt.predict_proba(scores.reshape(-1, 1))[:, 1]

    def _worst_env_brier(self, X_sel_std, y_sel, E_sel, w, b, platt):
        p1 = self._p1(X_sel_std @ w + b, platt)
        worst = 0.0
        for e in np.unique(E_sel):
            m = E_sel == e
            worst = max(worst, float(np.mean((p1[m] - y_sel[m]) ** 2)))
        return worst

    # ---- sklearn API -------------------------------------------------------------

    def fit(self, X, y, E):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        E = np.asarray(E)
        assert X.shape[1] == len(self.features), "X columns must match `features`"
        assert not (set(self.movable_features) & set(self.causal_features))
        self.classes_ = np.array([0, 1])
        self.movable_mask_ = np.array(
            [1.0 if f in self.movable_features else 0.0 for f in self.features]
        )

        X_std = self._standardize_fit(X)
        s = 1.0 - 2.0 * y  # +1 if Y=0 (favorable), -1 if Y=1

        # per-environment fit / selection split (selection set is held out)
        rng = np.random.default_rng(self.random_state)
        fit_idx, sel_idx = [], []
        for e in np.unique(E):
            idx_e = np.where(E == e)[0]
            rng.shuffle(idx_e)
            n_sel = max(1, int(round(self.val_frac * len(idx_e))))
            sel_idx.append(idx_e[:n_sel])
            fit_idx.append(idx_e[n_sel:])
        fit_idx = np.concatenate(fit_idx)
        sel_idx = np.concatenate(sel_idx)

        # alpha_0 from an initial ERM on the fit split
        self.alpha0_ = self._calibrate_alpha0(X_std[fit_idx], s[fit_idx])

        # h soft labels on the fit split (alpha-independent -> compute once)
        h_lab_fit = self._fit_h_labels(X_std[fit_idx], y[fit_idx])

        # grid search: worst-environment Brier on the held-out selection split
        self.selection_ = []
        best_alpha, best_worst = None, np.inf
        for m in self.alpha_mults:
            alpha = self.alpha0_ * m
            f = self._fit_linear(
                X_std[fit_idx], s[fit_idx], alpha, h_lab_fit, strategic=True
            )
            w = f.weight.detach().numpy().ravel()
            b = float(f.bias.detach().item())
            platt = self._fit_platt(X_std[fit_idx] @ w + b, y[fit_idx])
            worst = self._worst_env_brier(
                X_std[sel_idx], y[sel_idx], E[sel_idx], w, b, platt
            )
            self.selection_.append(
                {"alpha": alpha, "mult": m, "worst_env_brier": worst}
            )
            if worst < best_worst:
                best_worst, best_alpha = worst, alpha
        self.alpha_ = best_alpha

        # final refit on ALL training data at alpha* (fair vs. the all-data RF baselines)
        h_lab_all = self._fit_h_labels(X_std, y)
        f = self._fit_linear(X_std, s, best_alpha, h_lab_all, strategic=True)
        self._w = f.weight.detach().numpy().ravel()
        self._b = float(f.bias.detach().item())
        self._platt = self._fit_platt(X_std @ self._w + self._b, y)
        self.coef_ = self._w  # convenience for inspection
        return self

    def decision_score(self, X):
        """Linear score predicting ``s`` (high = favorable = low P(Y=1))."""
        return self._standardize(np.asarray(X, dtype=float)) @ self._w + self._b

    def predict_proba(self, X):
        scores = self.decision_score(X)
        return self._platt.predict_proba(scores.reshape(-1, 1))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
