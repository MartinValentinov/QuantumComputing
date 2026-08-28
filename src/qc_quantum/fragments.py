"""Model fragments for the functional groups the drawing code detects.

A drug molecule is far too large for quantum chemistry -- adagrasib has 43
heavy atoms, which is thousands of qubits in any honest encoding. Fragment
methods are the standard answer: the chemistry that decides reactivity is
local, so a small model system carrying the same functional group reproduces
the electronic behaviour that matters.

That is a real approximation and worth stating plainly. Acrolein stands in for
an acrylamide warhead; it has the same Michael-acceptor motif but not the same
substituents, so the absolute numbers belong to acrolein and only the ordering
between groups transfers to the drug.

Geometries come from ASE's reference set rather than being generated, because
this repo has no geometry optimiser and a bad geometry produces a confident
wrong energy.
"""

from __future__ import annotations

FRAGMENTS = {
    "acrylamide (covalent warhead)": {
        "formula": "CH3CHO",
        "label": "acrolein-like Michael acceptor",
        "why": "The warhead's reactivity comes from the conjugated C=C-C=O; "
               "this is the smallest system carrying it.",
    },
    "nitrile": {"formula": "CH3CN", "label": "acetonitrile",
                "why": "Carries the C≡N triple bond intact."},
    "amide": {"formula": "CH3CONH2", "label": "acetamide",
              "why": "The peptide-like C(=O)N unit."},
    "ester": {"formula": "HCOOCH3", "label": "methyl formate",
              "why": "The hydrolysable C(=O)O linkage."},
    "carboxylic acid": {"formula": "HCOOH", "label": "formic acid",
                        "why": "The ionisable acid group."},
    "benzene ring": {"formula": "C6H6", "label": "benzene",
                     "why": "The aromatic ring itself."},
    "pyridine ring": {"formula": "C5H5N", "label": "pyridine",
                      "why": "Aromatic ring with the basic nitrogen."},
    "azole ring": {"formula": "C4H4NH", "label": "pyrrole",
                   "why": "Five-membered aromatic heterocycle."},
    "furan ring": {"formula": "C4H4O", "label": "furan", "why": "Oxygen heterocycle."},
    "thiophene ring": {"formula": "C4H4S", "label": "thiophene", "why": "Sulfur heterocycle."},
    "alcohol": {"formula": "CH3OH", "label": "methanol", "why": "The hydroxyl group."},
    "ether": {"formula": "CH3OCH3", "label": "dimethyl ether", "why": "The C-O-C linkage."},
    "thiol": {"formula": "CH3SH", "label": "methanethiol",
              "why": "Also the model for the target cysteine a warhead attacks."},
    "basic amine": {"formula": "NH3", "label": "ammonia",
                    "why": "The protonatable nitrogen."},
}


def for_group(name: str) -> dict | None:
    """Find a model fragment for a detected functional group."""
    if name in FRAGMENTS:
        return {**FRAGMENTS[name], "group": name}
    lowered = name.lower()
    for key, spec in FRAGMENTS.items():
        if key.split(" (")[0] in lowered or lowered in key:
            return {**spec, "group": name}
    return None


def for_molecule(smiles: str) -> list[dict]:
    """Every group in a molecule that has a model fragment, most reactive first."""
    from qc_draw.features import detect
    from qc_model.smiles import parse

    seen: dict[str, dict] = {}
    for feature in detect(parse(smiles)):
        spec = for_group(feature.name)
        if spec and spec["formula"] not in seen:
            seen[spec["formula"]] = {**spec, "kind": feature.kind,
                                     "atoms": sorted(feature.atoms)}

    order = {"warhead": 0, "liability": 1, "group": 2}
    return sorted(seen.values(), key=lambda f: order.get(f.get("kind"), 3))
