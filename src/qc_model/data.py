"""Pulling a training set out of the research graph.

The graph already holds everything needed: compound structures from ChEMBL and
measured potencies against a target. This turns that into (X, y).

One decision worth naming: when a compound has been measured several times
against the same target, the values are averaged in log space. pChEMBL is
already a log scale, so the mean of pChEMBL values is the geometric mean of the
underlying concentrations -- which is the right way to average potencies.
Averaging raw IC50 would let a single weak measurement dominate.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from qc_model.fingerprint import featurize


def load_target_dataset(
    conn: sqlite3.Connection,
    target_chembl_id: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_descriptors: bool = True,
    min_measurements: int = 1,
    verbose: bool = True,
) -> dict:
    """Build (X, y, smiles) for one target from the graph."""
    rows = conn.execute(
        """SELECT c.chembl_id, c.smiles,
                  AVG(a.pchembl_value) AS y,
                  COUNT(*)             AS n_meas,
                  MAX(c.max_phase)     AS max_phase
           FROM activity a
           JOIN compound c ON c.chembl_id = a.chembl_id
           WHERE a.target_chembl_id = ?
             AND c.smiles IS NOT NULL
             AND a.pchembl_value IS NOT NULL
           GROUP BY c.chembl_id
           HAVING COUNT(*) >= ?
           ORDER BY c.chembl_id""",
        (target_chembl_id, min_measurements),
    ).fetchall()

    if not rows:
        raise ValueError(f"no usable activity data for {target_chembl_id}")

    features, labels, smiles, ids, failed = [], [], [], [], []
    for r in rows:
        vec = featurize(r["smiles"], radius=radius, n_bits=n_bits,
                        use_descriptors=use_descriptors)
        if vec is None:
            failed.append(r["chembl_id"])
            continue
        features.append(vec)
        labels.append(float(r["y"]))
        smiles.append(r["smiles"])
        ids.append(r["chembl_id"])

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)

    if verbose:
        print(f"  {len(rows)} compounds in graph, {len(y)} featurised, "
              f"{len(failed)} unparseable")
        print(f"  X {X.shape}  y range {y.min():.2f}–{y.max():.2f}  "
              f"mean {y.mean():.2f}  sd {y.std():.2f}")

    return {
        "X": X, "y": y, "smiles": smiles, "chembl_ids": ids,
        "n_failed": len(failed), "failed_ids": failed[:20],
        "target_chembl_id": target_chembl_id,
    }


def standardize(
    x_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Zero-mean, unit-variance using TRAINING statistics only.

    Computing the mean over the whole dataset leaks test information into
    training. It is a small leak and it always flatters the result.
    """
    mean = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return tuple((a - mean) / sd for a in (x_train, *others))
