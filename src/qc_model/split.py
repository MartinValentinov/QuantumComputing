"""Splitting data honestly.

A random train/test split on molecules produces a number that means nothing.
Medicinal chemistry datasets are full of *series*: twenty analogues of one
scaffold, differing by a methyl here and a fluorine there. Split randomly and
near-identical molecules land on both sides, so the model gets credit for
recognising something it has effectively already seen. Reported R² goes up;
real-world performance does not move.

A scaffold split groups molecules by their Bemis-Murcko framework -- the
molecule with its side chains stripped off, leaving rings and the linkers
between them -- and keeps whole groups together. Test scaffolds are then ones
the model has genuinely never seen, which is the question you actually care
about: does this work on new chemistry?

Expect scaffold-split scores to be substantially worse than random-split
scores. That gap is not a bug; it is the measurement working.
"""

from __future__ import annotations

import numpy as np

from qc_model.smiles import Mol, SmilesError, parse


def murcko_scaffold(mol: Mol) -> frozenset[tuple]:
    """The ring systems plus their connecting linkers.

    Implemented by repeatedly deleting terminal atoms that are not in a ring.
    What survives is the framework. Returned as a canonical set of typed
    atom/bond features rather than a SMILES string, since that is all the
    grouping needs and it avoids writing a SMILES generator.
    """
    keep = {a.index for a in mol.atoms}

    changed = True
    while changed:
        changed = False
        for i in list(keep):
            atom = mol.atoms[i]
            if atom.in_ring:
                continue
            live_neighbors = sum(1 for n in atom.neighbors if n in keep)
            if live_neighbors <= 1:
                keep.discard(i)
                changed = True

    if not keep:
        return frozenset()

    # Canonical description: multiset of (element, aromatic, degree-in-scaffold)
    # for atoms, and (element pair, order) for bonds. Independent of atom order.
    atom_feats = []
    for i in sorted(keep):
        a = mol.atoms[i]
        degree = sum(1 for n in a.neighbors if n in keep)
        atom_feats.append(("a", a.element, a.aromatic, degree, a.in_ring))

    bond_feats = []
    for b in mol.bonds:
        if b.a in keep and b.b in keep:
            pair = tuple(sorted((mol.atoms[b.a].element, mol.atoms[b.b].element)))
            bond_feats.append(("b", pair, b.order, b.in_ring))

    from collections import Counter

    return frozenset(Counter(atom_feats + bond_feats).items())


def scaffold_of(smiles: str) -> frozenset[tuple] | None:
    try:
        return murcko_scaffold(parse(smiles))
    except (SmilesError, ValueError):
        return None


def scaffold_split(
    smiles: list[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train / validation / test by scaffold group.

    Largest groups go to train first. This is the standard construction: it
    concentrates common chemistry in training and leaves the rare, unusual
    scaffolds for test, which is the harder and more honest arrangement.
    """
    groups: dict[frozenset, list[int]] = {}
    for i, smi in enumerate(smiles):
        key = scaffold_of(smi)
        groups.setdefault(key if key is not None else frozenset({("unparsed", i)}), []).append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

    n = len(smiles)
    n_train = int(fractions[0] * n)
    n_valid = int(fractions[1] * n)

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for group in ordered:
        if len(train) + len(group) <= n_train:
            train += group
        elif len(valid) + len(group) <= n_valid:
            valid += group
        else:
            test += group

    rng = np.random.default_rng(seed)
    for part in (train, valid, test):
        rng.shuffle(part)

    return np.array(train, dtype=int), np.array(valid, dtype=int), np.array(test, dtype=int)


def random_split(
    n: int, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kept only so the two can be compared. Do not report this one alone."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def split_report(smiles: list[str], train, valid, test) -> dict:
    """How much scaffold overlap survives. Should be zero for a scaffold split."""
    def scaffolds(idx):
        return {s for s in (scaffold_of(smiles[i]) for i in idx) if s}

    s_tr, s_va, s_te = scaffolds(train), scaffolds(valid), scaffolds(test)
    return {
        "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        "scaffolds_train": len(s_tr),
        "scaffolds_valid": len(s_va),
        "scaffolds_test": len(s_te),
        "overlap_train_test": len(s_tr & s_te),
        "overlap_train_valid": len(s_tr & s_va),
    }
