"""Using a trained model on new molecules.

The subtlety worth naming: standardisation statistics are part of the model.
Features must be centred with the *training* mean and scale, not with anything
computed from whatever is being predicted, so those numbers are saved beside
the weights and reloaded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qc_model.fingerprint import featurize
from qc_model.nn import MLP, load

MODEL_DIR = Path("data/models")


class Predictor:
    def __init__(self, weights: Path, meta: dict) -> None:
        self.meta = meta
        self.n_bits = int(meta["n_bits"])
        self.radius = int(meta["radius"])
        self.hidden = tuple(meta["hidden"])
        self.mean = np.asarray(meta["feature_mean"], dtype=np.float32)
        self.sd = np.asarray(meta["feature_sd"], dtype=np.float32)
        n_in = self.n_bits + 14
        self.model: MLP = load(weights, n_in, self.hidden, dropout=0.0)

        # Training fingerprints, for the applicability-domain check below.
        data = np.load(str(weights))
        self.train_fp = (
            np.unpackbits(data["train_fp"], axis=1)[:, : self.n_bits].astype(np.float32)
            if "train_fp" in data else None
        )
        self.train_popcount = (
            self.train_fp.sum(axis=1) if self.train_fp is not None else None
        )

    def nearest_similarity(self, fp: np.ndarray) -> float:
        """Highest Tanimoto similarity to anything in the training set.

        A model only knows the chemistry it was shown. Asked about ethanol, a
        kinase-inhibitor model will still return a plausible-looking potency,
        because nothing in its architecture can say "I have never seen anything
        like this". This number is what says it.
        """
        if self.train_fp is None:
            return float("nan")
        bits = (fp > 0).astype(np.float32)
        intersection = self.train_fp @ bits
        union = self.train_popcount + bits.sum() - intersection
        return float((intersection / np.maximum(union, 1.0)).max())

    def predict_smiles(self, smiles: str) -> dict | None:
        vec = featurize(smiles, radius=self.radius, n_bits=self.n_bits)
        if vec is None:
            return None
        x = ((vec - self.mean) / self.sd).astype(np.float32).reshape(1, -1)
        value = float(self.model.predict(x)[0])
        similarity = self.nearest_similarity(vec[: self.n_bits])

        # 0.4 Tanimoto on ECFP4 is the usual line between "same chemical
        # series" and "unrelated molecule". Below it, the number is an
        # extrapolation and should be presented as one.
        if similarity != similarity:
            domain = "unknown"
        elif similarity >= 0.4:
            domain = "in domain"
        elif similarity >= 0.25:
            domain = "edge of domain"
        else:
            domain = "out of domain"

        nanomolar = 10 ** (9 - value)
        return {
            "pchembl": round(value, 2),
            "nanomolar": round(nanomolar, 1) if nanomolar < 1e6 else None,
            "nearest_similarity": round(similarity, 3) if similarity == similarity else None,
            "domain": domain,
            "trustworthy": domain == "in domain",
            "target": self.meta.get("target"),
            "model_test_r2": self.meta.get("test_r2"),
            "model_test_rmse": self.meta.get("test_rmse"),
            "n_train": self.meta.get("n_train"),
        }


_cache: dict[str, Predictor] = {}


def available(model_dir: Path = MODEL_DIR) -> list[str]:
    if not model_dir.exists():
        return []
    return sorted(p.stem.replace(".meta", "") for p in model_dir.glob("*.meta.json"))


def get(symbol: str, model_dir: Path = MODEL_DIR) -> Predictor | None:
    key = symbol.upper()
    if key in _cache:
        return _cache[key]

    meta_path = model_dir / f"{key}.meta.json"
    weights = model_dir / f"{key}.npz"
    if not meta_path.exists() or not weights.exists():
        return None

    predictor = Predictor(weights, json.loads(meta_path.read_text()))
    _cache[key] = predictor
    return predictor
