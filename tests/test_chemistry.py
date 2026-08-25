"""Chemistry correctness tests.

Two properties matter, and both are silent when broken:

  1. Molecular formula -- if hydrogen counting is wrong, every fingerprint is
     wrong, and nothing downstream complains.
  2. Representation invariance -- the same molecule written two ways must give
     the same fingerprint. Without this the model learns to tell two spellings
     of benzene apart, which is worse than useless.
"""

from collections import Counter

import numpy as np
import pytest

from qc_model.fingerprint import featurize, morgan
from qc_model.smiles import parse

# name -> (equivalent SMILES spellings, expected molecular formula)
MOLECULES = {
    "benzene":       (["c1ccccc1", "C1=CC=CC=C1"], "C6H6"),
    "benzoic acid":  (["c1ccccc1C(=O)O", "OC(=O)c1ccccc1", "C1=CC=CC=C1C(O)=O"], "C7H6O2"),
    "pyridine":      (["n1ccccc1", "C1=CC=NC=C1"], "C5H5N"),
    "pyrrole":       (["[nH]1cccc1", "N1C=CC=C1"], "C4H5N"),
    "N-methylpyrrole": (["Cn1cccc1", "CN1C=CC=C1"], "C5H7N"),
    "furan":         (["o1cccc1", "O1C=CC=C1"], "C4H4O"),
    "imidazole":     (["c1cnc[nH]1", "N1C=NC=C1"], "C3H4N2"),
    "naphthalene":   (["c1ccc2ccccc2c1", "C1=CC2=CC=CC=C2C=C1"], "C10H8"),
    "indole":        (["c1ccc2[nH]ccc2c1", "C1=CC=C2NC=CC2=C1"], "C8H7N"),
    "quinoline":     (["c1ccc2ncccc2c1", "C1=CC=C2N=CC=CC2=C1"], "C9H7N"),
    "purine":        (["c1nc2[nH]cnc2cn1", "N1C=NC2=C1N=CN=C2"], "C5H4N4"),
    "anthracene":    (["c1ccc2cc3ccccc3cc2c1", "C1=CC=C2C=C3C=CC=CC3=CC2=C1"], "C14H10"),
    "caffeine":      (["CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"], "C8H10N4O2"),
    "2-pyridone":    (["O=c1cccc[nH]1", "O=C1C=CC=CN1"], "C5H5NO"),
    "aspirin":       (["CC(=O)Oc1ccccc1C(=O)O"], "C9H8O4"),
    "ibuprofen":     (["CC(C)Cc1ccc(cc1)C(C)C(=O)O"], "C13H18O2"),
    "glucose":       (["OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"], "C6H12O6"),
    "L-alanine":     (["C[C@H](N)C(=O)O"], "C3H7NO2"),
    "salt":          (["[Na+].[Cl-]"], "ClNa"),
}

# Rings that must NOT be perceived as aromatic, and how many aromatic atoms
# each is allowed. Tetralin has a genuine benzene ring fused to a saturated one.
NEGATIVE_CONTROLS = {
    "cyclohexane": ("C1CCCCC1", 0),
    "cyclohexene": ("C1=CCCCC1", 0),
    "cyclohexanone": ("O=C1CCCCC1", 0),
    "p-benzoquinone": ("O=C1C=CC(=O)C=C1", 0),
    "cyclohexadienone": ("O=C1C=CC=CC1", 0),
    "tetralin": ("C1CCc2ccccc2C1", 6),
}


def molecular_formula(mol) -> str:
    counts = Counter(a.element for a in mol.atoms)
    counts["H"] += sum(a.n_hydrogens for a in mol.atoms)
    order = ["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))
    return "".join(
        s + (str(counts[s]) if counts[s] > 1 else "") for s in order if counts.get(s)
    )


@pytest.mark.parametrize("name", list(MOLECULES))
def test_molecular_formula(name):
    spellings, expected = MOLECULES[name]
    for smi in spellings:
        assert molecular_formula(parse(smi)) == expected, f"{name} via {smi}"


@pytest.mark.parametrize("name", [k for k, v in MOLECULES.items() if len(v[0]) > 1])
def test_fingerprint_invariant_to_spelling(name):
    spellings, _ = MOLECULES[name]
    prints = [morgan(parse(s)) for s in spellings]
    for other in prints[1:]:
        assert np.array_equal(prints[0], other), f"{name} spellings disagree"


@pytest.mark.parametrize("name", list(NEGATIVE_CONTROLS))
def test_aromaticity_not_overreached(name):
    smi, expected = NEGATIVE_CONTROLS[name]
    got = sum(1 for a in parse(smi).atoms if a.aromatic)
    assert got == expected, f"{name}: {got} aromatic atoms, expected {expected}"


def test_atom_order_does_not_change_fingerprint():
    a = morgan(parse("CC(=O)Oc1ccccc1C(=O)O"))
    b = morgan(parse("O=C(O)c1ccccc1OC(C)=O"))
    assert np.array_equal(a, b)


def test_different_molecules_differ():
    assert not np.array_equal(morgan(parse("CCO")), morgan(parse("CCCO")))


def test_bad_smiles_returns_none():
    assert featurize("not a molecule") is None
    assert featurize("c1ccccc") is None          # unclosed ring
    assert featurize("CC(") is None              # unbalanced branch


def test_featurize_shape():
    v = featurize("CC(=O)Oc1ccccc1C(=O)O", n_bits=2048)
    assert v.shape == (2048 + 14,)
    assert v[:2048].sum() > 0
