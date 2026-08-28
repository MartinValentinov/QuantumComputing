"""Recognising functional groups in a molecular graph.

This is what turns "here is a picture" into "here is why it works". A
medicinal chemist reads a structure by spotting groups: an acrylamide means
the drug forms a permanent covalent bond; a carboxylic acid means it is
charged at blood pH and will struggle to cross membranes; a sulfonamide is a
classic hinge binder.

Rather than a general SMARTS engine, each group is a small explicit predicate
over the graph. That keeps every match auditable -- when the UI claims a
molecule has a covalent warhead, you can read the six lines that decided so.
"""

from __future__ import annotations

from dataclasses import dataclass

from qc_model.smiles import Mol


@dataclass
class Feature:
    name: str
    atoms: set[int]
    why: str
    kind: str = "group"


def _elem(mol: Mol, i: int) -> str:
    return mol.atoms[i].element


def _nb(mol: Mol, i: int) -> list[int]:
    return mol.atoms[i].neighbors


def _double_to(mol: Mol, i: int, element: str) -> list[int]:
    return [
        mol.other(b, i) for b in mol.bonds_of(i)
        if b.order == 2.0 and _elem(mol, mol.other(b, i)) == element
    ]


def _single_to(mol: Mol, i: int, element: str) -> list[int]:
    return [
        mol.other(b, i) for b in mol.bonds_of(i)
        if b.order == 1.0 and _elem(mol, mol.other(b, i)) == element
    ]


def detect(mol: Mol) -> list[Feature]:
    """Every group we can recognise, most pharmacologically interesting first."""
    found: list[Feature] = []
    seen: set[frozenset] = set()

    def add(name, atoms, why, kind="group"):
        key = frozenset(atoms)
        if key in seen:
            return
        seen.add(key)
        found.append(Feature(name, set(atoms), why, kind))

    for i, atom in enumerate(mol.atoms):
        el = atom.element

        if el == "C":
            oxo = _double_to(mol, i, "O")
            if oxo:
                o_double = oxo[0]
                n_single = _single_to(mol, i, "N")
                o_single = _single_to(mol, i, "O")

                if n_single:
                    n = n_single[0]
                    vinyl = [
                        c for c in _nb(mol, i)
                        if _elem(mol, c) == "C" and c != n
                        and any(
                            b.order == 2.0 and _elem(mol, mol.other(b, c)) == "C"
                            and not mol.atoms[mol.other(b, c)].aromatic
                            for b in mol.bonds_of(c)
                        )
                        and not mol.atoms[c].aromatic
                    ]
                    if vinyl:
                        add("acrylamide (covalent warhead)",
                            [i, o_double, n, *vinyl],
                            "An electrophilic alkene next to an amide. It forms a "
                            "permanent covalent bond to a cysteine in the binding "
                            "site, so the drug does not wash out — this is the "
                            "chemistry behind the KRAS G12C inhibitors.",
                            kind="warhead")
                    else:
                        add("amide", [i, o_double, n],
                            "The backbone linkage of peptides. Rigid, and both a "
                            "hydrogen-bond donor and acceptor, which is why it "
                            "appears in so many drugs that grip a protein.")
                elif o_single:
                    o = o_single[0]
                    if mol.atoms[o].n_hydrogens > 0:
                        add("carboxylic acid", [i, o_double, o],
                            "Deprotonated and negatively charged at blood pH. Good "
                            "for solubility and for salt bridges, bad for crossing "
                            "membranes — carboxylic acids rarely reach the brain.")
                    else:
                        add("ester", [i, o_double, o],
                            "Readily cleaved by esterases in blood and liver. Often "
                            "deliberate: a prodrug that releases the active molecule.")
                elif not atom.aromatic:
                    add("ketone", [i, o_double],
                        "A hydrogen-bond acceptor and a point of metabolic attack.")

            triple_n = [
                mol.other(b, i) for b in mol.bonds_of(i)
                if b.order == 3.0 and _elem(mol, mol.other(b, i)) == "N"
            ]
            if triple_n:
                add("nitrile", [i, triple_n[0]],
                    "Small, linear, strongly polar. Often used to pick up a "
                    "hydrogen bond in a pocket too tight for anything bigger.")

        if el == "S":
            oxo = _double_to(mol, i, "O")
            n_single = _single_to(mol, i, "N")
            if len(oxo) >= 2 and n_single:
                add("sulfonamide", [i, *oxo[:2], n_single[0]],
                    "A strong, geometrically distinctive acceptor pair. Common in "
                    "kinase inhibitors and the original antibacterial sulfa drugs.")

        if el == "O" and atom.n_hydrogens > 0 and len(_nb(mol, i)) == 1:
            parent = _nb(mol, i)[0]
            if not _double_to(mol, parent, "O"):
                aromatic = mol.atoms[parent].aromatic
                add("phenol" if aromatic else "hydroxyl", [i, parent],
                    "Donates and accepts hydrogen bonds. Raises solubility, and is "
                    "a common site of metabolic conjugation."
                    if not aromatic else
                    "An aromatic hydroxyl. Strong hydrogen-bond donor, and a "
                    "frequent target for the liver's conjugating enzymes.")

        if el == "N" and not atom.aromatic and atom.charge == 0:
            heavy = _nb(mol, i)
            carbonyl_neighbor = any(_double_to(mol, c, "O") for c in heavy if _elem(mol, c) == "C")
            sulfonyl_neighbor = any(_elem(mol, c) == "S" for c in heavy)
            if not carbonyl_neighbor and not sulfonyl_neighbor and len(heavy) <= 3:
                add("basic amine", [i],
                    "Protonated and positively charged at blood pH. Improves "
                    "solubility and can form a salt bridge, but too many of these "
                    "keeps a molecule out of the brain.")

        if el in ("F", "Cl", "Br", "I"):
            add(f"{el} substituent", [i],
                "Halogens block metabolism at that position and tune electron "
                "density. Fluorine especially: small enough to sit where a "
                "hydrogen was, but far harder for the liver to remove.")

    from qc_model.smiles import find_rings

    for ring in find_rings(mol):
        elements = sorted(_elem(mol, i) for i in ring)
        aromatic = all(mol.atoms[i].aromatic for i in ring)
        size = len(ring)

        if size == 6 and not aromatic and elements.count("N") == 2 and elements.count("C") == 4:
            add("piperazine", ring,
                "A solubilising ring with two basic nitrogens. One of the most "
                "used fragments in medicinal chemistry — it is how a greasy "
                "molecule is made injectable.")
        elif size == 6 and not aromatic and elements.count("N") == 1 and elements.count("O") == 1:
            add("morpholine", ring,
                "Adds solubility without much basicity. A standard way to fix a "
                "compound that binds well but will not dissolve.")
        elif size == 6 and aromatic and elements.count("N") == 0:
            add("benzene ring", ring,
                "A flat, rigid, greasy scaffold. Usually the part that stacks "
                "against aromatic residues in the binding site.")
        elif size == 6 and aromatic and elements.count("N") >= 1:
            add(f"{'pyridine' if elements.count('N') == 1 else 'diazine'} ring", ring,
                "An aromatic ring with a built-in hydrogen-bond acceptor. The "
                "nitrogen frequently makes the key contact to a kinase hinge.")
        elif size == 5 and aromatic and elements.count("N") >= 1:
            add("azole ring", ring,
                "A small aromatic heterocycle that both donates and accepts. "
                "Extremely common in kinase inhibitors.")

    order = {"warhead": 0, "liability": 1, "group": 2}
    found.sort(key=lambda f: (order.get(f.kind, 3), -len(f.atoms)))
    return found


def lipinski(mol: Mol) -> dict:
    """Rule-of-5 assessment, computed from the graph rather than looked up."""
    from qc_draw.render import ELEMENT_COLOR
    from qc_model.fingerprint import ATOMIC_MASS, LOGP_CONTRIB

    mw = sum(ATOMIC_MASS.get(a.element, 12.0) + a.n_hydrogens * 1.008 for a in mol.atoms)
    logp = sum(LOGP_CONTRIB.get(a.element, 0.0) for a in mol.atoms)
    hbd = sum(1 for a in mol.atoms if a.element in ("N", "O") and a.n_hydrogens > 0)
    hba = sum(1 for a in mol.atoms if a.element in ("N", "O"))

    violations = []
    if mw > 500:
        violations.append(f"molecular weight {mw:.0f} exceeds 500")
    if logp > 5:
        violations.append(f"estimated logP {logp:.1f} exceeds 5")
    if hbd > 5:
        violations.append(f"{hbd} hydrogen-bond donors exceeds 5")
    if hba > 10:
        violations.append(f"{hba} hydrogen-bond acceptors exceeds 10")

    return {
        "mw": round(mw, 1), "clogp": round(logp, 2), "hbd": hbd, "hba": hba,
        "violations": violations,
        "verdict": (
            "drug-like by Lipinski's rule of 5" if not violations
            else f"{len(violations)} Lipinski violation(s) — oral absorption may suffer"
        ),
    }
