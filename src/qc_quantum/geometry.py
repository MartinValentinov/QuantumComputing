"""Molecular geometry: build it, hold it, hand it to a driver."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Molecule:
    """A molecule as a list of atoms with 3D coordinates in Angstrom."""

    name: str
    atoms: list[tuple[str, float, float, float]]
    charge: int = 0
    spin: int = 0

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def to_xyz(self) -> str:
        """Standard .xyz format. Consumed by py3Dmol, VMD, most viewers."""
        lines = [str(self.n_atoms), self.name]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in self.atoms]
        return "\n".join(lines) + "\n"

    def to_pyscf(self) -> str:
        """PySCF's `atom=` string format."""
        return "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in self.atoms)


def from_formula(formula: str, charge: int = 0, spin: int = 0) -> Molecule:
    """Build from ASE's library of known geometries (H2O, NH3, CH4, ...).

    Only covers small molecules ASE ships geometries for. For anything
    else, use `from_xyz` or generate coordinates with RDKit.
    """
    from ase.build import molecule as ase_molecule

    atoms = ase_molecule(formula)
    return Molecule(
        name=formula,
        atoms=[(a.symbol, float(a.x), float(a.y), float(a.z)) for a in atoms],
        charge=charge,
        spin=spin,
    )


def from_xyz(path: str, charge: int = 0, spin: int = 0) -> Molecule:
    """Read a standard .xyz file."""
    with open(path) as fh:
        lines = fh.read().strip().splitlines()

    n = int(lines[0].strip())
    name = lines[1].strip() or path
    atoms = []
    for line in lines[2 : 2 + n]:
        sym, x, y, z = line.split()[:4]
        atoms.append((sym, float(x), float(y), float(z)))

    return Molecule(name=name, atoms=atoms, charge=charge, spin=spin)


def view_3d(mol: Molecule, width: int = 400, height: int = 400):
    """Render in a Jupyter notebook. No-op outside one."""
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol.to_xyz(), "xyz")
    view.setStyle({"sphere": {"scale": 0.3}, "stick": {"radius": 0.1}})
    view.zoomTo()
    return view.show()
