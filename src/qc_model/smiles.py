"""A SMILES parser: text -> molecular graph.

SMILES is a linear notation for a branching, cyclic graph. The parser is a
single left-to-right pass with two pieces of state: a stack for branches, and
a table of open ring-closure digits.

    CC(=O)Oc1ccccc1C(=O)O          aspirin
    ^^                             two chained atoms -> a bond between them
      (       )                    branch: push on '(', pop on ')'
       =O                          bond symbol applies to the NEXT atom
         c1ccccc1                  '1' opens a ring closure, the second closes it
         ^                         lowercase = aromatic

Implicit hydrogens are the subtle part and they matter: hydrogen count is an
input to the fingerprint, so getting it wrong silently degrades every
downstream prediction. Aromatic bonds are counted as order 1.5, which makes
benzene carbons come out at 1 H and fusion carbons at 0 -- the right answer
without needing full ring perception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ORGANIC = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"]
AROMATIC_ORGANIC = ["c", "n", "o", "p", "s", "b"]

VALENCES: dict[str, tuple[int, ...]] = {
    "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5),
    "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,),
}

BOND_ORDER = {"-": 1.0, "=": 2.0, "#": 3.0, "$": 4.0, ":": 1.5, "/": 1.0, "\\": 1.0}

_BRACKET = re.compile(
    r"\[(?P<iso>\d+)?"
    r"(?P<sym>[A-Za-z][a-z]?|\*)"
    r"(?P<chiral>@{1,2}(?:TH|AL|SP|TB|OH)?\d*)?"
    r"(?:H(?P<h>\d*))?"
    r"(?P<charge>(?:[+-]\d+)|(?:\++)|(?:-+))?"
    r"(?::(?P<map>\d+))?\]"
)


class SmilesError(ValueError):
    pass


@dataclass
class Atom:
    index: int
    symbol: str
    aromatic: bool = False
    charge: int = 0
    n_hydrogens: int = 0
    explicit_h: bool = False
    in_ring: bool = False
    neighbors: list[int] = field(default_factory=list)

    @property
    def element(self) -> str:
        """Canonical element symbol. `symbol` is already normalised on parse --
        upper() would turn Cl into CL and Na into NA."""
        return self.symbol


@dataclass
class Bond:
    a: int
    b: int
    order: float = 1.0
    aromatic: bool = False
    in_ring: bool = False


@dataclass
class Mol:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    smiles: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def bonds_of(self, i: int) -> list[Bond]:
        return [b for b in self.bonds if b.a == i or b.b == i]

    def other(self, bond: Bond, i: int) -> int:
        return bond.b if bond.a == i else bond.a


def parse(smiles: str) -> Mol:
    """Parse a SMILES string into a molecular graph. Raises SmilesError."""
    mol = Mol(smiles=smiles)
    branch_stack: list[int] = []
    ring_open: dict[str, tuple[int, float | None]] = {}
    prev: int | None = None
    pending_bond: float | None = None
    pending_aromatic_bond = False

    i, n = 0, len(smiles)
    while i < n:
        ch = smiles[i]

        if ch == "(":
            if prev is None:
                raise SmilesError("branch opened before any atom")
            branch_stack.append(prev)
            i += 1
            continue

        if ch == ")":
            if not branch_stack:
                raise SmilesError("unbalanced ')'")
            prev = branch_stack.pop()
            i += 1
            continue

        if ch in BOND_ORDER:
            pending_bond = BOND_ORDER[ch]
            pending_aromatic_bond = ch == ":"
            i += 1
            continue

        if ch == ".":
            prev = None
            i += 1
            continue

        if ch.isdigit() or ch == "%":
            if ch == "%":
                label, i = smiles[i + 1 : i + 3], i + 3
            else:
                label, i = ch, i + 1
            if prev is None:
                raise SmilesError(f"ring closure {label} before any atom")

            if label in ring_open:
                partner, order = ring_open.pop(label)
                use = pending_bond or order or 1.0
                arom = mol.atoms[partner].aromatic and mol.atoms[prev].aromatic
                _add_bond(mol, partner, prev, use, arom or pending_aromatic_bond, ring=True)
            else:
                ring_open[label] = (prev, pending_bond)
            pending_bond, pending_aromatic_bond = None, False
            continue

        if ch == "[":
            m = _BRACKET.match(smiles, i)
            if not m:
                raise SmilesError(f"bad bracket atom at position {i}")
            atom = _bracket_atom(m, len(mol.atoms))
            i = m.end()
        else:
            symbol = next(
                (s for s in ORGANIC if smiles.startswith(s, i)),
                next((s for s in AROMATIC_ORGANIC if smiles.startswith(s, i)), None),
            )
            if symbol is None:
                raise SmilesError(f"unexpected character {ch!r} at position {i}")
            atom = Atom(
                index=len(mol.atoms),
                symbol=symbol.capitalize(),
                aromatic=symbol.islower(),
            )
            i += len(symbol)

        mol.atoms.append(atom)

        if prev is not None:
            arom = (
                pending_aromatic_bond
                or (mol.atoms[prev].aromatic and atom.aromatic and pending_bond is None)
            )
            order = pending_bond if pending_bond is not None else (1.5 if arom else 1.0)
            _add_bond(mol, prev, atom.index, order, arom)

        prev = atom.index
        pending_bond, pending_aromatic_bond = None, False

    if ring_open:
        raise SmilesError(f"unclosed ring bond(s): {sorted(ring_open)}")
    if branch_stack:
        raise SmilesError("unbalanced '('")

    _assign_hydrogens(mol)
    _mark_rings(mol)
    perceive_aromaticity(mol)
    return mol


def _bracket_atom(m: re.Match, index: int) -> Atom:
    sym = m.group("sym")
    charge_s = m.group("charge") or ""
    if charge_s.startswith(("+", "-")) and len(charge_s) > 1 and charge_s[1:].isdigit():
        charge = int(charge_s[0] + charge_s[1:])
    elif charge_s:
        charge = len(charge_s) * (1 if charge_s[0] == "+" else -1)
    else:
        charge = 0

    h_group = m.group("h")
    n_h = (int(h_group) if h_group else 1) if h_group is not None else 0

    return Atom(
        index=index,
        symbol=sym.capitalize(),
        aromatic=sym.islower(),
        charge=charge,
        n_hydrogens=n_h,
        explicit_h=True,
    )


def _add_bond(mol: Mol, a: int, b: int, order: float, aromatic: bool, ring: bool = False) -> None:
    if a == b:
        raise SmilesError("self-bond")
    mol.bonds.append(Bond(a=a, b=b, order=1.5 if aromatic else order,
                          aromatic=aromatic, in_ring=ring))
    mol.atoms[a].neighbors.append(b)
    mol.atoms[b].neighbors.append(a)


def _assign_hydrogens(mol: Mol) -> None:
    """Fill in implicit hydrogens for atoms written without brackets.

    Aromatic atoms need their own rule. Counting each aromatic bond as 1.5
    works for benzene but breaks on a trisubstituted aromatic nitrogen
    (N-methylpyrrole, caffeine): 1.5+1.5+1.0 = 4.0 overshoots nitrogen's
    valence of 3 and invents a hydrogen.

    Counting the sigma framework instead -- one per aromatic bond -- and then
    adding a single pi bond gets every case right, provided the pi bond is
    omitted for atoms that donate a lone pair to the ring instead. A nitrogen,
    oxygen or sulfur with three sigma connections has no room for the pi bond,
    which is exactly the pyrrole-type case.
    """
    for atom in mol.atoms:
        if atom.explicit_h:
            continue
        valences = VALENCES.get(atom.symbol)
        if not valences:
            atom.n_hydrogens = 0
            continue

        bonds = mol.bonds_of(atom.index)
        if atom.aromatic:
            sigma = sum(1.0 if b.aromatic else b.order for b in bonds)
            donates_lone_pair = atom.element in ("N", "O", "S", "Se") and sigma >= 3
            used = sigma if donates_lone_pair else sigma + 1.0
        else:
            used = sum(b.order for b in bonds)

        target = next((v for v in valences if v >= used - atom.charge), valences[-1])
        atom.n_hydrogens = max(0, int(target + atom.charge - used + 0.5))


def _mark_rings(mol: Mol) -> None:
    """Flag ring membership. A bond is in a ring iff removing it leaves its
    endpoints connected -- i.e. it is not a bridge."""
    adjacency: dict[int, list[tuple[int, int]]] = {a.index: [] for a in mol.atoms}
    for bi, b in enumerate(mol.bonds):
        adjacency[b.a].append((b.b, bi))
        adjacency[b.b].append((b.a, bi))

    for bi, bond in enumerate(mol.bonds):
        if bond.in_ring:
            continue
        seen, stack = {bond.a}, [bond.a]
        while stack:
            cur = stack.pop()
            if cur == bond.b:
                bond.in_ring = True
                break
            for nxt, edge in adjacency[cur]:
                if edge != bi and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

    for bond in mol.bonds:
        if bond.in_ring:
            mol.atoms[bond.a].in_ring = True
            mol.atoms[bond.b].in_ring = True


AROMATIC_ELEMENTS = {"C", "N", "O", "S", "Se"}
LONE_PAIR_DONORS = {"N", "O", "S", "Se"}


def find_rings(mol: Mol, max_size: int = 8) -> list[list[int]]:
    """Smallest ring through each ring bond. Not a true SSSR, but it finds
    every ring that matters for aromaticity in drug-like molecules."""
    adjacency: dict[int, list[tuple[int, int]]] = {a.index: [] for a in mol.atoms}
    for bi, b in enumerate(mol.bonds):
        adjacency[b.a].append((b.b, bi))
        adjacency[b.b].append((b.a, bi))

    seen: set[frozenset[int]] = set()
    rings: list[list[int]] = []

    for bi, bond in enumerate(mol.bonds):
        if not bond.in_ring:
            continue
        prev: dict[int, int] = {bond.a: -1}
        queue = [bond.a]
        while queue:
            cur = queue.pop(0)
            if cur == bond.b:
                break
            for nxt, edge in adjacency[cur]:
                if edge != bi and nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
        if bond.b not in prev:
            continue

        path, node = [], bond.b
        while node != -1:
            path.append(node)
            node = prev[node]
        if len(path) > max_size:
            continue
        key = frozenset(path)
        if key not in seen:
            seen.add(key)
            rings.append(path)
    return rings


def perceive_aromaticity(mol: Mol) -> int:
    """Mark Kekule rings as aromatic. Returns the number of rings converted.

    Two phases, and the order matters. Converting rings as they are tested
    breaks fused systems: aromatising the first ring of naphthalene rewrites
    the shared bond from order 2.0 to 1.5, so when the second ring is tested
    its atoms no longer appear to carry a double bond and it fails. Every ring
    is therefore judged against the original Kekule bond orders, and only then
    are the conversions applied.
    """
    rings = find_rings(mol)
    ring_atoms: set[int] = {i for r in rings for i in r}

    candidates = []
    for ring in rings:
        if len(ring) not in (5, 6):
            continue
        if all(mol.atoms[i].aromatic for i in ring):
            continue
        if any(mol.atoms[i].element not in AROMATIC_ELEMENTS for i in ring):
            continue

        ring_set = set(ring)

        has_donor = any(
            mol.atoms[i].element in LONE_PAIR_DONORS
            and not any(
                b.order == 2.0 and mol.other(b, i) in ring_set
                for b in mol.bonds_of(i)
            )
            for i in ring
        )

        ok = True
        for i in ring:
            atom = mol.atoms[i]
            if atom.element in LONE_PAIR_DONORS:
                continue
            if any(b.order == 2.0 and mol.other(b, i) in ring_atoms
                   for b in mol.bonds_of(i)):
                continue
            if has_donor and any(
                b.order == 2.0 and mol.atoms[mol.other(b, i)].element in ("O", "N", "S")
                for b in mol.bonds_of(i)
            ):
                continue
            ok = False
            break
        if ok:
            candidates.append(ring_set)

    for ring_set in candidates:
        for i in ring_set:
            mol.atoms[i].aromatic = True
        for b in mol.bonds:
            if b.a in ring_set and b.b in ring_set:
                b.order, b.aromatic = 1.5, True

    return len(candidates)


