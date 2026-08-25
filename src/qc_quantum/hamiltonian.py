"""Molecule -> electronic structure problem -> qubit Hamiltonian.

This is the part that decides how many qubits you need, which is the
single most important number in the whole quantum-chemistry story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qc_quantum.geometry import Molecule

MapperName = Literal["jordan_wigner", "parity", "bravyi_kitaev"]


@dataclass
class QubitProblem:
    """Everything downstream solvers need, plus the cost numbers."""

    molecule: Molecule
    basis: str
    qubit_op: object  # qiskit.quantum_info.SparsePauliOp
    n_qubits: int
    n_pauli_terms: int
    nuclear_repulsion: float
    hf_energy: float
    problem: object  # qiskit_nature ElectronicStructureProblem

    def summary(self) -> str:
        return (
            f"{self.molecule.name} / {self.basis}: "
            f"{self.n_qubits} qubits, {self.n_pauli_terms} Pauli terms, "
            f"E_nuc={self.nuclear_repulsion:.6f} Ha"
        )


def build_hamiltonian(
    mol: Molecule,
    basis: str = "sto3g",
    mapper: MapperName = "parity",
    freeze_core: bool = True,
) -> QubitProblem:
    """Run PySCF, map to qubits, report the cost.

    `parity` mapping with two-qubit reduction saves 2 qubits over
    Jordan-Wigner for closed-shell systems -- worth it, since qubit
    count is the binding constraint.
    """
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )
    from qiskit_nature.second_q.transformers import FreezeCoreTransformer
    from qiskit_nature.units import DistanceUnit

    driver = PySCFDriver(
        atom=mol.to_pyscf(),
        basis=basis,
        charge=mol.charge,
        spin=mol.spin,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    hf_energy = float(problem.reference_energy)

    if freeze_core:
        problem = FreezeCoreTransformer().transform(problem)

    second_q_op = problem.hamiltonian.second_q_op()

    if mapper == "jordan_wigner":
        qmapper = JordanWignerMapper()
    elif mapper == "bravyi_kitaev":
        qmapper = BravyiKitaevMapper()
    else:
        qmapper = ParityMapper(num_particles=problem.num_particles)

    qubit_op = qmapper.map(second_q_op)

    return QubitProblem(
        molecule=mol,
        basis=basis,
        qubit_op=qubit_op,
        n_qubits=qubit_op.num_qubits,
        n_pauli_terms=len(qubit_op),
        nuclear_repulsion=float(problem.nuclear_repulsion_energy),
        hf_energy=hf_energy,
        problem=problem,
    )
