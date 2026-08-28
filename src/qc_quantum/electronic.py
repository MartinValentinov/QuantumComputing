"""Electronic descriptors from quantum chemistry, computed on a quantum circuit.

This is where quantum computing earns its keep in this project, and it is worth
being precise about why. Quantum machine learning is contested -- the fair
kernel test in `fair_test.py` shows the quantum kernel losing to a classical
one on identical inputs. Simulating electronic structure is not contested. It
is a quantum system being modelled by a quantum system, which is the original
argument for building these machines at all, and it needs no speedup claim to
be the right tool.

What gets computed:

    I  ionisation energy   E(N-1) - E(N)
    A  electron affinity   E(N) - E(N+1)
    mu chemical potential  -(I + A) / 2
    eta hardness           I - A
    omega electrophilicity mu^2 / (2 eta)

Three separate ground-state calculations -- neutral, cation, anion -- rather
than reading orbital energies off a single one. That is the rigorous route:
Koopmans' theorem gets I from the HOMO for free but ignores how the remaining
electrons relax once one leaves, and for the small strained systems that make
good warheads the relaxation is exactly the interesting part.

Electrophilicity predicts how readily a molecule accepts electron density,
which for a covalent inhibitor is the property that decides whether its warhead
attaches to the target cysteine at a useful rate. Too low and it never reacts;
too high and it reacts with everything in the cell, which is how covalent drugs
become toxic.

Everything runs in an active space of a few frontier orbitals. The core
electrons are chemically inert here and carrying them would cost dozens of
qubits to describe something that does not change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

HARTREE_TO_EV = 27.211386


@dataclass
class ElectronicResult:
    name: str
    formula: str
    n_qubits: int = 0
    n_pauli_terms: int = 0
    energies: dict = field(default_factory=dict)
    ionisation_eV: float | None = None
    affinity_eV: float | None = None
    chemical_potential_eV: float | None = None
    hardness_eV: float | None = None
    electrophilicity_eV: float | None = None
    method: str = ""
    seconds: float = 0.0
    error: str | None = None


DEFAULT_BASIS = "6-31+g*"


def _solve(atom: str, charge: int, spin: int, active_electrons: tuple[int, int],
           active_orbitals: int, use_vqe: bool,
           basis: str = DEFAULT_BASIS) -> tuple[float, int, int]:
    """Ground-state energy of one charge state in the active space.

    `active_electrons` must be an explicit (alpha, beta) pair. Passing a single
    total is only valid for closed shells: the cation and anion carry an odd
    electron count, which cannot be split evenly, and the transformer rejects
    it rather than guessing.
    """
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import ParityMapper
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    problem = PySCFDriver(atom=atom, basis=basis, charge=charge, spin=spin).run()
    reduced = ActiveSpaceTransformer(
        num_electrons=active_electrons, num_spatial_orbitals=active_orbitals
    ).transform(problem)

    mapper = ParityMapper(num_particles=reduced.num_particles)
    qubit_op = mapper.map(reduced.hamiltonian.second_q_op())
    shifts = sum(float(v) for v in (reduced.hamiltonian.constants or {}).values())

    if not use_vqe:
        from qiskit_algorithms import NumPyMinimumEigensolver

        value = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
        return float(value.eigenvalue.real) + shifts, qubit_op.num_qubits, len(qubit_op)

    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

    initial = HartreeFock(reduced.num_spatial_orbitals, reduced.num_particles, mapper)
    ansatz = UCCSD(reduced.num_spatial_orbitals, reduced.num_particles, mapper,
                   initial_state=initial)
    solver = VQE(estimator=StatevectorEstimator(), ansatz=ansatz,
                 optimizer=SLSQP(maxiter=200))
    solver.initial_point = [0.0] * ansatz.num_parameters
    value = solver.compute_minimum_eigenvalue(qubit_op)
    return float(value.eigenvalue.real) + shifts, qubit_op.num_qubits, len(qubit_op)


def descriptors(atom: str, name: str, formula: str = "",
                active_electrons: int = 4, active_orbitals: int = 4,
                use_vqe: bool = True, basis: str = DEFAULT_BASIS) -> ElectronicResult:
    """Ionisation, affinity, hardness and electrophilicity for one molecule.

    `use_vqe=False` swaps in exact diagonalisation of the same active-space
    Hamiltonian. Same physics, same qubit count, no variational optimisation --
    useful as a reference for checking that VQE actually converged.

    The basis defaults to 6-31+G* rather than a minimal one because the anion
    calculation needs diffuse functions: in STO-3G formaldehyde's electron
    affinity came out at -8.2 eV, which is a basis-set artefact rather than
    chemistry. Widening the basis costs nothing in qubits -- the active space
    fixes that at six however many orbitals the basis provides -- which is
    precisely what active spaces are for.
    """
    t0 = time.perf_counter()
    result = ElectronicResult(name=name, formula=formula,
                              method=("VQE/UCCSD" if use_vqe else "exact") + f" / CAS({active_electrons},{active_orbitals}) / {basis}")
    try:
        half = active_electrons // 2
        neutral, n_qubits, n_terms = _solve(
            atom, 0, 0, (half, half), active_orbitals, use_vqe, basis)
        cation, _, _ = _solve(
            atom, 1, 1, (half, half - 1), active_orbitals, use_vqe, basis)
        anion, _, _ = _solve(
            atom, -1, 1, (half + 1, half), active_orbitals, use_vqe, basis)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.seconds = time.perf_counter() - t0
        return result

    result.n_qubits = n_qubits
    result.n_pauli_terms = n_terms
    result.energies = {"neutral": neutral, "cation": cation, "anion": anion}

    ionisation = (cation - neutral) * HARTREE_TO_EV
    affinity = (neutral - anion) * HARTREE_TO_EV
    mu = -(ionisation + affinity) / 2
    eta = ionisation - affinity

    result.ionisation_eV = round(ionisation, 3)
    result.affinity_eV = round(affinity, 3)
    result.chemical_potential_eV = round(mu, 3)
    result.hardness_eV = round(eta, 3)
    result.electrophilicity_eV = round(mu * mu / (2 * eta), 3) if abs(eta) > 1e-6 else None
    result.seconds = round(time.perf_counter() - t0, 2)
    return result


def from_ase(formula: str, **kwargs) -> ElectronicResult:
    """Run the calculation on a molecule from ASE's reference geometries."""
    from ase.build import molecule as ase_molecule

    atoms = ase_molecule(formula)
    atom_str = "; ".join(f"{a.symbol} {a.x} {a.y} {a.z}" for a in atoms)
    return descriptors(atom_str, name=kwargs.pop("name", formula),
                       formula=formula, **kwargs)
