"""VQE, with the chemistry-aware ansatz rather than a generic one.

A note on why this is here at all:

  The original script used TwoLocal(ry, cx, reps=2). That is a
  hardware-efficient ansatz -- it ignores chemistry entirely, and its
  gradients vanish exponentially with qubit count (a "barren plateau").
  It gives plausible answers for H2/H2O and then silently stops working.

  UCCSD instead builds excitations out of the Hartree-Fock reference,
  so it starts in a physically sensible place and stays interpretable.
  It has far more parameters and is slower, but it is the honest
  baseline. Keep `ansatz="two_local"` available only for comparison.
"""

from __future__ import annotations

import time
from typing import Literal

from qc_quantum.hamiltonian import QubitProblem
from qc_quantum.solvers.classical import EnergyResult

AnsatzName = Literal["uccsd", "two_local"]


def solve_vqe(
    qp: QubitProblem,
    ansatz: AnsatzName = "uccsd",
    maxiter: int = 300,
    seed: int = 42,
) -> EnergyResult:
    """Minimise <psi|H|psi> on a noiseless statevector simulator.

    Noiseless is intentional: it isolates the ansatz's expressiveness
    from hardware noise. Real-hardware numbers will be worse, and that
    gap is itself the interesting measurement.
    """
    import numpy as np
    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import COBYLA, SLSQP

    problem = qp.problem

    if ansatz == "uccsd":
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
        from qiskit_nature.second_q.mappers import ParityMapper

        mapper = ParityMapper(num_particles=problem.num_particles)
        initial_state = HartreeFock(
            problem.num_spatial_orbitals, problem.num_particles, mapper
        )
        circuit = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=initial_state,
        )
        optimizer = SLSQP(maxiter=maxiter)
    else:
        from qiskit.circuit.library import TwoLocal

        circuit = TwoLocal(
            qp.n_qubits, rotation_blocks="ry", entanglement_blocks="cx", reps=2
        )
        optimizer = COBYLA(maxiter=maxiter)

    np.random.seed(seed)
    estimator = StatevectorEstimator()

    vqe = VQE(estimator=estimator, ansatz=circuit, optimizer=optimizer)
    if ansatz == "uccsd":
        vqe.initial_point = [0.0] * circuit.num_parameters

    t0 = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(qp.qubit_op)
    wall = time.perf_counter() - t0

    electronic = float(result.eigenvalue.real)

    # The eigenvalue is only the *active space* energy. Recovering a total
    # energy means adding back every constant the problem carries: nuclear
    # repulsion, and -- crucially -- the frozen-core shift, which is large
    # (tens of Hartree for anything with a second-row atom). Adding nuclear
    # repulsion alone silently produces a number that is wrong by exactly the
    # core energy, and it looks plausible enough to miss.
    total, shifts = _total_energy(problem, electronic, qp.nuclear_repulsion)

    return EnergyResult(
        method=f"VQE/{ansatz}",
        energy=total,
        wall_seconds=wall,
        converged=True,
        detail={
            "active_space_energy": electronic,
            "energy_shifts": shifts,
            "n_parameters": int(circuit.num_parameters),
            "n_qubits": qp.n_qubits,
            "n_pauli_terms": qp.n_pauli_terms,
            "optimizer_evals": int(getattr(result, "cost_function_evals", 0) or 0),
        },
    )


def _total_energy(problem, electronic: float, nuclear_repulsion: float):
    """Assemble a total energy from an active-space eigenvalue.

    Prefers qiskit-nature's own `interpret`, which knows about every
    transformer applied to the problem. Falls back to summing the
    Hamiltonian's constants, and only as a last resort to nuclear repulsion
    alone -- which is correct only when no core was frozen.
    """
    try:
        interpreted = problem.interpret(
            type("_R", (), {"eigenvalue": complex(electronic, 0.0)})()
        )
        return float(interpreted.total_energies[0].real), {"via": "interpret"}
    except Exception:
        pass

    constants = dict(getattr(problem.hamiltonian, "constants", {}) or {})
    if constants:
        return electronic + sum(float(v) for v in constants.values()), constants

    return electronic + nuclear_repulsion, {"nuclear_repulsion": nuclear_repulsion}
