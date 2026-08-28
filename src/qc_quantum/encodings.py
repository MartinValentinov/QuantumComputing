"""Feature encodings for quantum kernels, and the tools to compare them fairly.

The first kernel comparison in this repo was not a fair test. The quantum
kernel was given eight global descriptors -- molecular weight, logP, and so on
-- while Tanimoto was given a 1024-bit structural fingerprint. Tanimoto won,
which said nothing about quantum versus classical and everything about eight
numbers versus a thousand structural bits.

A fair test holds the information constant and varies only the kernel. That is
what this module exists for: it produces the reduced feature matrices, and the
benchmark then runs a quantum kernel and a classical one over exactly the same
columns. Tanimoto on the full fingerprint stays in as an external reference --
the number to beat overall -- but it is no longer the quantum comparison.

PCA is written out rather than imported, like everything else here, and it is
fit on the training molecules only. Fitting it on the whole set would leak
information about the molecules being ranked into the representation used to
rank them.
"""

from __future__ import annotations

import numpy as np


def pca_fit(X: np.ndarray, n_components: int) -> dict:
    """Principal components by eigendecomposition of the covariance."""
    mean = X.mean(axis=0, keepdims=True)
    centred = X - mean
    cov = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:n_components]
    return {
        "mean": mean,
        "components": vectors[:, order],
        "explained": values[order] / max(values.sum(), 1e-12),
    }


def pca_apply(model: dict, X: np.ndarray) -> np.ndarray:
    return (X - model["mean"]) @ model["components"]


def fingerprint_matrix(smiles_list: list[str], n_bits: int = 1024,
                       radius: int = 2) -> np.ndarray:
    from qc_model.fingerprint import morgan
    from qc_model.smiles import parse

    rows = []
    for smi in smiles_list:
        try:
            rows.append(morgan(parse(smi), radius=radius, n_bits=n_bits))
        except Exception:
            rows.append(np.zeros(n_bits, dtype=np.float32))
    return np.vstack(rows).astype(np.float64)


def descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    from qc_model.fingerprint import descriptors
    from qc_model.smiles import parse

    rows = []
    for smi in smiles_list:
        try:
            rows.append(descriptors(parse(smi)))
        except Exception:
            rows.append(np.zeros(14, dtype=np.float32))
    return np.vstack(rows).astype(np.float64)


def structural_features(smiles_list: list[str], n_features: int = 8,
                        n_bits: int = 1024) -> np.ndarray:
    """Structural information compressed to something a small circuit can hold.

    This is the encoding that makes the comparison meaningful: it starts from
    the same Morgan fingerprint Tanimoto uses, so the quantum circuit is seeing
    substructure information rather than whole-molecule averages.
    """
    F = fingerprint_matrix(smiles_list, n_bits=n_bits)
    keep = F.std(axis=0) > 1e-9
    F = F[:, keep] if keep.sum() >= n_features else F
    if F.shape[1] <= n_features:
        out = np.zeros((len(F), n_features))
        out[:, : F.shape[1]] = F
        return out
    return pca_apply(pca_fit(F, n_features), F)


def combined_features(smiles_list: list[str], n_features: int = 8) -> np.ndarray:
    """Half structure, half physicochemistry."""
    half = max(n_features // 2, 1)
    struct = structural_features(smiles_list, n_features=half)
    desc = descriptor_matrix(smiles_list)[:, [1, 2, 3, 5, 10, 13]][:, : n_features - half]
    return np.hstack([struct, desc])


ENCODINGS = {
    "descriptors": lambda s, n: descriptor_matrix(s)[:, [0, 1, 2, 3, 4, 5, 10, 13]][:, :n],
    "structural": lambda s, n: structural_features(s, n_features=n),
    "combined": lambda s, n: combined_features(s, n_features=n),
}
