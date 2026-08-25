"""Molecular graph -> fixed-length feature vector.

Two featurisations, both written out rather than imported:

**Circular fingerprints (ECFP/Morgan).** Each atom starts with an identifier
derived from its own properties. Each round, an atom's new identifier hashes
together its old one and its neighbours' -- so after `radius` rounds an
identifier encodes everything within `radius` bonds. Collecting identifiers
from every round and folding them modulo `n_bits` gives a bit vector where a
set bit means "this substructure is present somewhere".

The hash must be stable across processes, which rules out Python's built-in
`hash()` for strings -- it is salted per interpreter, so a model trained in one
run would see different bits in the next. crc32 is deterministic and fast.

**Descriptors.** A handful of whole-molecule numbers -- size, polarity,
flexibility. Weak alone, but they carry information fingerprints do not (a
fingerprint knows which fragments are present, not how big the molecule is).
"""

from __future__ import annotations

import zlib

import numpy as np

from qc_model.smiles import Mol, parse

ATOMIC_NUMBER = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Fe": 26,
    "Zn": 30, "Se": 34, "Br": 35, "I": 53,
}

# Crude per-element contributions, in the spirit of Crippen's atomic logP.
# Not calibrated -- they exist to give the model a polarity signal, and the
# model is free to ignore them.
LOGP_CONTRIB = {"C": 0.30, "N": -0.60, "O": -0.55, "S": 0.25, "F": 0.35,
                "Cl": 0.60, "Br": 0.75, "I": 0.90, "P": -0.20}
TPSA_CONTRIB = {"N": 12.0, "O": 17.0, "S": 12.0, "P": 13.0}

ATOMIC_MASS = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
               "F": 18.998, "Na": 22.99, "Si": 28.09, "P": 30.974, "S": 32.06,
               "Cl": 35.45, "K": 39.098, "Br": 79.904, "I": 126.90, "Se": 78.97}


def _h(*parts: int) -> int:
    """Deterministic hash of a tuple of ints."""
    payload = b"".join(int(p).to_bytes(8, "little", signed=True) for p in parts)
    return zlib.crc32(payload)


def _initial_identifiers(mol: Mol) -> list[int]:
    """Daylight-style atom invariants: what an atom is, ignoring context."""
    ids = []
    for atom in mol.atoms:
        bonds = mol.bonds_of(atom.index)
        heavy_degree = len(bonds)
        valence = int(sum(b.order for b in bonds))
        ids.append(
            _h(
                heavy_degree,
                valence,
                ATOMIC_NUMBER.get(atom.element, 0),
                atom.charge,
                atom.n_hydrogens,
                int(atom.in_ring),
                int(atom.aromatic),
            )
        )
    return ids


def morgan(mol: Mol, radius: int = 2, n_bits: int = 2048, counts: bool = False) -> np.ndarray:
    """ECFP-style fingerprint. `radius=2` corresponds to ECFP4."""
    vec = np.zeros(n_bits, dtype=np.float32)
    ids = _initial_identifiers(mol)

    for ident in ids:
        idx = ident % n_bits
        vec[idx] = vec[idx] + 1 if counts else 1.0

    for layer in range(1, radius + 1):
        new_ids = []
        for atom in mol.atoms:
            # Sorting makes the identifier independent of neighbour order,
            # which is what makes the fingerprint invariant to how the SMILES
            # happened to be written.
            env = sorted(
                (int(b.order * 2), ids[mol.other(b, atom.index)])
                for b in mol.bonds_of(atom.index)
            )
            flat: list[int] = [layer, ids[atom.index]]
            for order, nid in env:
                flat += [order, nid]
            new_ids.append(_h(*flat))

        for ident in new_ids:
            idx = ident % n_bits
            vec[idx] = vec[idx] + 1 if counts else 1.0
        ids = new_ids

    return vec


ROTATABLE_EXCLUDE = {"H"}


def descriptors(mol: Mol) -> np.ndarray:
    """Whole-molecule properties. Small, interpretable, and complementary."""
    n_heavy = mol.n_atoms
    mw = sum(
        ATOMIC_MASS.get(a.element, 12.0) + a.n_hydrogens * 1.008 for a in mol.atoms
    )
    logp = sum(LOGP_CONTRIB.get(a.element, 0.0) for a in mol.atoms)
    tpsa = sum(
        TPSA_CONTRIB.get(a.element, 0.0) - (2.0 if a.n_hydrogens else 0.0)
        for a in mol.atoms
        if a.element in TPSA_CONTRIB
    )
    hbd = sum(1 for a in mol.atoms if a.element in ("N", "O") and a.n_hydrogens > 0)
    hba = sum(1 for a in mol.atoms if a.element in ("N", "O"))
    aromatic_atoms = sum(1 for a in mol.atoms if a.aromatic)
    ring_atoms = sum(1 for a in mol.atoms if a.in_ring)
    n_rings = len([b for b in mol.bonds if b.in_ring]) - ring_atoms + 1 if ring_atoms else 0
    halogens = sum(1 for a in mol.atoms if a.element in ("F", "Cl", "Br", "I"))
    charge = sum(a.charge for a in mol.atoms)

    # A rotatable bond: single, acyclic, and not to a terminal atom.
    rotatable = sum(
        1 for b in mol.bonds
        if b.order == 1.0 and not b.in_ring
        and len(mol.bonds_of(b.a)) > 1 and len(mol.bonds_of(b.b)) > 1
    )
    sp3ish = sum(
        1 for a in mol.atoms
        if a.element == "C" and not a.aromatic
        and sum(bb.order for bb in mol.bonds_of(a.index)) + a.n_hydrogens >= 4
    )

    return np.array(
        [
            n_heavy, mw / 100.0, logp, tpsa / 10.0, hbd, hba,
            aromatic_atoms, ring_atoms, max(n_rings, 0), halogens,
            rotatable, charge, sp3ish / max(n_heavy, 1),
            aromatic_atoms / max(n_heavy, 1),
        ],
        dtype=np.float32,
    )


DESCRIPTOR_NAMES = [
    "n_heavy", "mw/100", "clogp", "tpsa/10", "hbd", "hba", "n_aromatic",
    "n_ring_atoms", "n_rings", "n_halogen", "n_rotatable", "charge",
    "frac_sp3", "frac_aromatic",
]


def featurize(
    smiles: str, radius: int = 2, n_bits: int = 2048, use_descriptors: bool = True
) -> np.ndarray | None:
    """SMILES -> feature vector. Returns None if the string will not parse."""
    try:
        mol = parse(smiles)
    except Exception:
        return None
    if mol.n_atoms == 0:
        return None

    fp = morgan(mol, radius=radius, n_bits=n_bits)
    if not use_descriptors:
        return fp
    return np.concatenate([fp, descriptors(mol)])
