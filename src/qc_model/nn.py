"""A small feed-forward network, written out rather than imported.

Forward pass, loss, backward pass and the optimiser are all here in numpy.
There is no autograd: each layer states its own gradient explicitly, which is
the point -- backprop stops being magic once you have written the six lines
that implement it.

Architecture is deliberately plain: a couple of hidden layers with ReLU, plus
dropout, trained with Adam and early stopping on a validation set. On a few
thousand molecules that is the right size. Anything deeper memorises the
training scaffolds and scores worse on the split that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Linear:
    """y = xW + b, and the gradients thereof."""

    n_in: int
    n_out: int
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    rng_seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.rng_seed)
        self.W = rng.normal(0.0, np.sqrt(2.0 / self.n_in), (self.n_in, self.n_out)).astype(np.float32)
        self.b = np.zeros(self.n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask

    def params(self):
        return []


class Dropout:
    """Inverted dropout: scale at train time so inference needs no change."""

    def __init__(self, p: float, seed: int = 0) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0:
            return x
        self._mask = (self.rng.random(x.shape) > self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0:
            return grad
        return grad * self._mask

    def params(self):
        return []


class Adam:
    """Adam, written out. Momentum on the gradient and on its square, both
    bias-corrected because they start at zero and would otherwise bias early
    steps toward zero."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def step(self, params) -> None:
        self.t += 1
        for p, g in params:
            key = id(p)
            if key not in self.state:
                self.state[key] = (np.zeros_like(p), np.zeros_like(p))
            m, v = self.state[key]

            grad = g + self.weight_decay * p if self.weight_decay else g
            m[:] = self.b1 * m + (1 - self.b1) * grad
            v[:] = self.b2 * v + (1 - self.b2) * grad * grad

            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (512, 128),
                 dropout: float = 0.25, seed: int = 0) -> None:
        self.layers: list = []
        prev = n_in
        for i, h in enumerate(hidden):
            self.layers.append(Linear(prev, h, rng_seed=seed + i))
            self.layers.append(ReLU())
            self.layers.append(Dropout(dropout, seed=seed + 100 + i))
            prev = h
        self.layers.append(Linear(prev, 1, rng_seed=seed + 99))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x.ravel()

    def backward(self, grad: np.ndarray) -> None:
        g = grad.reshape(-1, 1)
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def params(self):
        out = []
        for layer in self.layers:
            out += layer.params()
        return out

    def train(self, mode: bool = True) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = mode

    def predict(self, x: np.ndarray, batch: int = 512) -> np.ndarray:
        self.train(False)
        out = [self.forward(x[i : i + batch]) for i in range(0, len(x), batch)]
        self.train(True)
        return np.concatenate(out) if out else np.array([])


def mse_loss(pred: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error and its gradient with respect to `pred`."""
    diff = pred - y
    return float((diff ** 2).mean()), (2.0 / len(y)) * diff


def fit(
    model: MLP,
    x_train: np.ndarray, y_train: np.ndarray,
    x_valid: np.ndarray, y_valid: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 25,
    seed: int = 0,
    log_every: int = 10,
    verbose: bool = True,
) -> dict:
    """Train with early stopping on validation RMSE.

    The best weights are kept, not the last ones -- validation loss usually
    turns upward well before training loss does, and stopping without
    restoring means shipping an already-overfit model.
    """
    opt = Adam(lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    n = len(x_train)

    history: list[dict] = []
    best = {"rmse": float("inf"), "epoch": -1, "weights": None}
    since_improved = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        model.train(True)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = model.forward(x_train[idx])
            loss, grad = mse_loss(pred, y_train[idx])
            model.backward(grad)
            opt.step(model.params())
            epoch_loss += loss
            n_batches += 1

        val_pred = model.predict(x_valid)
        val_rmse = float(np.sqrt(((val_pred - y_valid) ** 2).mean()))
        train_rmse = float(np.sqrt(epoch_loss / max(n_batches, 1)))
        history.append({"epoch": epoch, "train_rmse": train_rmse, "valid_rmse": val_rmse})

        if val_rmse < best["rmse"] - 1e-5:
            best = {
                "rmse": val_rmse,
                "epoch": epoch,
                "weights": [(p.copy()) for p, _ in model.params()],
            }
            since_improved = 0
        else:
            since_improved += 1

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            marker = " *" if best["epoch"] == epoch else ""
            print(f"  epoch {epoch:>3}  train {train_rmse:.4f}  valid {val_rmse:.4f}{marker}",
                  flush=True)

        if since_improved >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no improvement for {patience})")
            break

    if best["weights"] is not None:
        for (p, _), saved in zip(model.params(), best["weights"]):
            p[:] = saved

    return {"history": history, "best_epoch": best["epoch"], "best_valid_rmse": best["rmse"]}


def save(model: MLP, path, meta: dict | None = None,
         train_fingerprints: "np.ndarray | None" = None) -> None:
    """Weights plus whatever the caller needs to reproduce the features.

    Feature settings travel with the weights on purpose: a model trained on
    2048-bit radius-2 fingerprints silently produces nonsense if it is later
    fed 1024-bit ones, and nothing raises.
    """
    import json
    from pathlib import Path

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"p{i}": p for i, (p, _) in enumerate(model.params())}
    if train_fingerprints is not None:
        arrays["train_fp"] = np.packbits(
            (train_fingerprints > 0).astype(np.uint8), axis=1
        )
    np.savez_compressed(path, **arrays)
    path.with_suffix(".meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))


def load(path, n_in: int, hidden: tuple[int, ...], dropout: float = 0.0) -> "MLP":
    import numpy as np

    model = MLP(n_in, hidden=hidden, dropout=dropout, seed=0)
    data = np.load(str(path) if str(path).endswith(".npz") else f"{path}.npz")
    for i, (p, _) in enumerate(model.params()):
        p[:] = data[f"p{i}"]
    model.train(False)
    return model
