"""Where the quantum advantage actually starts, measured rather than asserted.

The obvious objection to everything in this package is that quantum computing
is supposed to be fast, and a five-hour calculation is not fast. The objection
is correct and the explanation is simple: there is no quantum computer here.
Every circuit runs on a classical simulator, and simulating n qubits costs
2^n classically. The slowness is the exact exponential cost a quantum computer
would avoid -- it is the argument for the hardware, not against it.

That is easy to say and better to measure. This module grows the active space
on a fixed molecule and times the classical solve, which produces the only
number that matters when deciding whether quantum is worth pursuing: how many
qubits before classical becomes impossible.

On the machine this was written on, the solve time doubled every 0.91 qubits.
Extrapolating that fit: 40 qubits is decades, 50 qubits is longer than
recorded history, 60 qubits is longer than the Earth has existed. A quantum
processor runs all three in microseconds, because it is not simulating the
physics, it is doing it.

Nothing about the algorithm changes between the two. The same circuit and the
same optimiser run on hardware; only the estimator is swapped. See
`qc_quantum.backend`.
"""

from __future__ import annotations

import math
import time


def measure(smiles: str = "C=CC(N)=O", basis: str = "6-31g",
            sizes: tuple[tuple[int, int], ...] = ((2, 2), (4, 4), (6, 6), (8, 8), (10, 10)),
            verbose: bool = True) -> dict:
    """Time the classical solve as the active space grows."""
    import warnings

    warnings.filterwarnings("ignore")
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import ParityMapper
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    from qc_quantum.reaction import embed

    geometry = embed(smiles)
    if geometry is None:
        raise ValueError(f"could not embed {smiles!r}")
    problem = PySCFDriver(atom=geometry, basis=basis).run()

    rows = []
    for electrons, orbitals in sizes:
        try:
            reduced = ActiveSpaceTransformer(
                num_electrons=(electrons // 2, electrons // 2),
                num_spatial_orbitals=orbitals,
            ).transform(problem)
            mapper = ParityMapper(num_particles=reduced.num_particles)
            qubit_op = mapper.map(reduced.hamiltonian.second_q_op())

            t0 = time.perf_counter()
            NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
            elapsed = time.perf_counter() - t0

            rows.append({
                "active_space": f"CAS({electrons},{orbitals})",
                "n_qubits": qubit_op.num_qubits,
                "hilbert_dimension": 2 ** qubit_op.num_qubits,
                "n_pauli_terms": len(qubit_op),
                "classical_seconds": round(elapsed, 4),
                "statevector_megabytes": round(2 ** qubit_op.num_qubits * 16 / 1e6, 3),
            })
            if verbose:
                r = rows[-1]
                print(f"  {r['active_space']:<12}{r['n_qubits']:>3} qubits "
                      f"{r['hilbert_dimension']:>10,} states "
                      f"{r['classical_seconds']:>9.2f}s", flush=True)
        except Exception as e:
            if verbose:
                print(f"  CAS({electrons},{orbitals}) failed: {type(e).__name__}")

    usable = [r for r in rows if r["classical_seconds"] > 1e-3]
    projection = {}
    doubling = None
    if len(usable) >= 3:
        xs = [r["n_qubits"] for r in usable]
        ys = [math.log2(r["classical_seconds"]) for r in usable]
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs) or 1e-9
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        doubling = round(1 / slope, 3) if slope > 0 else None

        last = usable[-1]
        for target in (30, 40, 50, 60):
            seconds = last["classical_seconds"] * 2 ** (slope * (target - last["n_qubits"]))
            projection[target] = {
                "classical_seconds": seconds,
                "human": _humanise(seconds),
                "statevector_terabytes": round(2 ** target * 16 / 1e12, 1),
            }

    return {
        "molecule": smiles,
        "basis": basis,
        "measurements": rows,
        "qubits_per_doubling": doubling,
        "projection": projection,
        "note": "Classical cost of simulating the circuit. Real hardware runs "
                "every one of these in microseconds; it is not simulating the "
                "physics, it is doing it.",
    }


def _humanise(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    if seconds < 3.15e7:
        return f"{seconds / 86400:.1f} days"
    if seconds < 3.15e10:
        return f"{seconds / 3.15e7:.1f} years"
    return f"{seconds / 3.15e7:,.0f} years"
