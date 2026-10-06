"""Covalent warhead detection and thia-Michael adduct construction.

These exist because of a bug that produced no error anywhere. Both the
detection SMARTS and the reaction template pinned the alpha carbon as [CH],
meaning exactly one hydrogen, so every alpha substituted Michael acceptor was
invisible. Adagrasib carries a 2 fluoroacrylamide. The result was a pipeline
that reported "no covalent warhead, this compound binds reversibly" for the
approved covalent KRAS inhibitor it was written to showcase, while the build
sheet on the same page correctly labelled the acrylamide.

Nothing raised. The UI simply skipped its most interesting step.
"""

import pytest

from qc_quantum.reaction import (
    BENCHMARK_WARHEADS,
    analyse_covalent,
    reaction_energy,
    warhead_model,
)

ADAGRASIB = (
    "C=C(F)C(=O)N1CCN(c2nc(OC[C@@H]3CCCN3C)nc3c2CCN(c2cccc4cccc(Cl)c24)C3)"
    "C[C@@H]1CC#N"
)

ACCEPTORS = {
    "2-fluoroacrylamide": "C=C(F)C(N)=O",
    "acrylamide": "C=CC(N)=O",
    "methacrylamide": "C=C(C)C(N)=O",
    "acrolein": "C=CC=O",
    "methyl acrylate": "C=CC(=O)OC",
    "vinyl sulfone": "C=CS(C)(=O)=O",
    "acrylonitrile": "C=CC#N",
}

NON_ACCEPTORS = {
    "benzene": "c1ccccc1",
    "methanol": "CO",
    "propane": "CCC",
    "styrene": "C=Cc1ccccc1",
    "plain amide": "CC(=O)NC",
    "saturated nitrile": "CCC#N",
}


@pytest.mark.parametrize("name", list(ACCEPTORS))
def test_acceptors_are_detected(name):
    assert warhead_model(ACCEPTORS[name]) is not None, name


@pytest.mark.parametrize("name", list(NON_ACCEPTORS))
def test_non_acceptors_are_refused(name):
    assert warhead_model(NON_ACCEPTORS[name]) is None, name


@pytest.mark.parametrize("name,smiles", BENCHMARK_WARHEADS)
def test_benchmark_set_intact(name, smiles):
    assert warhead_model(smiles) is not None, name


def test_adagrasib_warhead_is_found_and_excised():
    """The regression. Detection, excision and adduct must all survive."""
    result = analyse_covalent(ADAGRASIB, methods=())
    assert result["has_warhead"] is True
    assert result["warhead_model"]["smiles"] == "C=C(F)C(N)=O"
    assert result["adduct_smiles"] == "CSCC(F)C(N)=O"
    assert result["error"] is None


def test_substituent_survives_into_the_adduct():
    """The alpha substituent must still be there after the addition.

    Writing the product alpha carbon as [CH2:2] would force two hydrogens onto
    a carbon that only has room for one, which is the mistake that a naive fix
    to the reactant pattern invites.
    """
    assert reaction_energy("C=C(F)C(N)=O", methods=()).adduct_smiles == "CSCC(F)C(N)=O"
    assert reaction_energy("C=C(C)C(N)=O", methods=()).adduct_smiles == "CSCC(C)C(N)=O"
    assert reaction_energy("C=CC(N)=O", methods=()).adduct_smiles == "CSCCC(N)=O"


def test_electronics_order_the_reactivity():
    """An electron withdrawing alpha substituent should make the alkene more
    electrophilic, and a donating one less.

    This is the chemistry check on the fix rather than a plumbing check: it is
    precisely why adagrasib carries a fluorine there.
    """
    energy = {
        name: reaction_energy(smiles, methods=("hf",)).delta_e_kcal["hf"]
        for name, smiles in (
            ("fluoro", "C=C(F)C(N)=O"),
            ("plain", "C=CC(N)=O"),
            ("methyl", "C=C(C)C(N)=O"),
        )
    }
    assert all(v < 0 for v in energy.values()), energy
    assert energy["fluoro"] < energy["plain"] < energy["methyl"], energy
