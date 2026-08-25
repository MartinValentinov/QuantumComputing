"""Quantum chemistry kernel: molecule -> Hamiltonian -> energy.

Deliberately small in scope. This package answers one question well:
"for this molecule, what energy do quantum and classical methods give,
 and at what cost?"

It is a benchmarking and research module, not the design engine.
"""

from qc_quantum.geometry import Molecule, from_formula, from_xyz
from qc_quantum.hamiltonian import build_hamiltonian

__all__ = ["Molecule", "from_formula", "from_xyz", "build_hamiltonian"]
