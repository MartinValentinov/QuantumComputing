"""Evaluation. Several numbers, because any one of them can flatter a model."""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((pred - y) ** 2).mean()))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    """Fraction of variance explained. Zero means no better than the mean;
    negative means actively worse than the mean, which does happen."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def pearson(pred: np.ndarray, y: np.ndarray) -> float:
    """Zero when either side is constant -- a flat prediction has no
    correlation, and numpy would return NaN with a divide warning."""
    if pred.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(pred, y)[0, 1])
    return 0.0 if np.isnan(r) else r


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation -- often what actually matters, since the practical
    question is which compounds to make next, not their exact potency."""
    def rank(a):
        order = a.argsort()
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a))
        return r

    return pearson(rank(pred), rank(y))


def evaluate(pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "rmse": rmse(pred, y),
        "mae": mae(pred, y),
        "r2": r2(pred, y),
        "pearson": pearson(pred, y),
        "spearman": spearman(pred, y),
    }


def ridge_regression(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Closed-form ridge, as the baseline the network has to beat.

    If a linear model on the same features does as well, the network is not
    earning its complexity and should not be shipped.
    """
    n_features = x_train.shape[1]
    xb = np.hstack([x_train, np.ones((len(x_train), 1), dtype=np.float32)])
    a = xb.T @ xb + alpha * np.eye(n_features + 1, dtype=np.float32)
    a[-1, -1] -= alpha                     # do not penalise the intercept
    return np.linalg.solve(a, xb.T @ y_train)


def ridge_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1), dtype=np.float32)]) @ w
