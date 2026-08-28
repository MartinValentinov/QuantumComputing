"""Does the quantum feature map add anything a classical kernel cannot?

The comparison that matters holds the input constant. Both kernels get the
same feature matrix; the only difference is how similarity is computed from
it. Anything else measures representation, not quantumness.

Scoring uses the same neighbour test as elsewhere: take each molecule as a
query, look at its k nearest neighbours under a kernel, and measure how far
their measured potencies sit from the query's. A kernel that captures
something real puts similar potencies together, so lower is better. Tanimoto
over the full fingerprint is reported alongside as the external reference --
the number any method has to beat to be worth using -- but it is not the
quantum comparison, because it sees far more information.
"""

from __future__ import annotations

import math
import statistics as st
import time

import numpy as np

from qc_quantum.encodings import ENCODINGS


def _angles(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X
    lo, hi = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return (X - lo) / span * math.pi


def quantum_matrix(X: np.ndarray, reps: int = 2, entanglement: str = "full") -> np.ndarray:
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector

    n = X.shape[1]
    circuit = zz_feature_map(feature_dimension=n, reps=reps, entanglement=entanglement)
    states = np.zeros((len(X), 2 ** n), dtype=complex)
    for i, row in enumerate(_angles(X)[:, :n]):
        states[i] = Statevector.from_instruction(circuit.assign_parameters(row)).data
    return np.abs(states.conj() @ states.T) ** 2


def rbf_matrix(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    Z = _angles(X)
    sq = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    g = gamma if gamma is not None else 1.0 / max(Z.shape[1], 1)
    return np.exp(-g * sq)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Z = _angles(X)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return (Z / norm) @ (Z / norm).T


def tanimoto_matrix(smiles_list: list[str], n_bits: int = 1024) -> np.ndarray:
    from qc_quantum.encodings import fingerprint_matrix

    F = fingerprint_matrix(smiles_list, n_bits=n_bits)
    inter = F @ F.T
    sizes = F.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    return np.divide(inter, np.where(union == 0, 1, union))


def neighbour_gap(matrix: np.ndarray, labels: list[float], k: int = 5) -> float:
    """Mean |potency difference| between each molecule and its k neighbours."""
    gaps = []
    for i in range(len(labels)):
        row = matrix[i].copy()
        row[i] = -np.inf
        for j in np.argsort(-row)[:k]:
            gaps.append(abs(labels[int(j)] - labels[i]))
    return float(st.mean(gaps)) if gaps else 0.0


def random_gap(labels: list[float], trials: int = 600) -> float:
    import random

    rng = random.Random(0)
    n = len(labels)
    if n < 2:
        return 0.0
    return float(st.mean(
        abs(labels[rng.randrange(n)] - labels[rng.randrange(n)]) for _ in range(trials)
    ))


def run(smiles_list: list[str], labels: list[float], n_qubits: int = 8,
        k: int = 5, reps: int = 2) -> dict:
    """Every encoding x every kernel, on identical inputs."""
    results = []
    for enc_name, enc in ENCODINGS.items():
        X = enc(smiles_list, n_qubits)
        row: dict = {"encoding": enc_name, "n_features": int(X.shape[1])}

        t0 = time.perf_counter()
        qm = quantum_matrix(X, reps=reps)
        row["quantum"] = round(neighbour_gap(qm, labels, k), 4)
        row["quantum_seconds"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        row["rbf"] = round(neighbour_gap(rbf_matrix(X), labels, k), 4)
        row["rbf_seconds"] = round(time.perf_counter() - t0, 4)

        row["cosine"] = round(neighbour_gap(cosine_matrix(X), labels, k), 4)
        row["quantum_wins"] = bool(row["quantum"] < min(row["rbf"], row["cosine"]))
        row["margin_vs_rbf"] = round(row["rbf"] - row["quantum"], 4)
        results.append(row)

    return {
        "n_molecules": len(smiles_list),
        "n_qubits": n_qubits,
        "reps": reps,
        "per_encoding": results,
        "tanimoto_reference": round(neighbour_gap(tanimoto_matrix(smiles_list), labels, k), 4),
        "random_floor": round(random_gap(labels), 4),
    }
