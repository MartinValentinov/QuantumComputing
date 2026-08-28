"""Run quantum and classical side by side and report the gap.

This is the actual deliverable of the quantum half of the project:
a defensible table of "where is the crossover, and what does it cost".
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from qc_quantum.geometry import Molecule, from_formula
from qc_quantum.hamiltonian import build_hamiltonian
from qc_quantum.solvers.classical import EnergyResult, solve_classical
from qc_quantum.solvers.vqe import solve_vqe

CHEMICAL_ACCURACY_HA = 0.0016


def benchmark(
    mol: Molecule,
    basis: str = "sto3g",
    ansatz: str = "uccsd",
    reference: str = "FCI",
) -> dict:
    qp = build_hamiltonian(mol, basis=basis)
    classical = solve_classical(mol, basis=basis)
    quantum = solve_vqe(qp, ansatz=ansatz)

    all_results: list[EnergyResult] = [*classical, quantum]

    ref = next(
        (r for r in all_results if r.method == reference and r.energy == r.energy),
        None,
    )
    rows = []
    for r in all_results:
        err = None if ref is None else r.energy - ref.energy
        rows.append(
            {
                "method": r.method,
                "energy_hartree": r.energy,
                "error_vs_ref": err,
                "within_chemical_accuracy": (
                    None if err is None else abs(err) < CHEMICAL_ACCURACY_HA
                ),
                "wall_seconds": r.wall_seconds,
                "detail": r.detail,
            }
        )

    return {
        "molecule": mol.name,
        "n_atoms": mol.n_atoms,
        "basis": basis,
        "n_qubits": qp.n_qubits,
        "n_pauli_terms": qp.n_pauli_terms,
        "reference": reference,
        "results": rows,
    }


def format_table(report: dict) -> str:
    head = (
        f"{report['molecule']}  ({report['n_atoms']} atoms, {report['basis']}, "
        f"{report['n_qubits']} qubits, {report['n_pauli_terms']} Pauli terms)"
    )
    lines = [head, "-" * len(head)]
    lines.append(f"{'method':<14}{'E (Ha)':>16}{'err vs ref':>14}{'chem acc':>10}{'sec':>9}")
    for r in report["results"]:
        e = r["energy_hartree"]
        err = r["error_vs_ref"]
        ok = r["within_chemical_accuracy"]
        lines.append(
            f"{r['method']:<14}"
            f"{e:>16.8f}"
            f"{(f'{err:+.6f}' if err is not None else '--'):>14}"
            f"{('yes' if ok else '--' if ok is None else 'no'):>10}"
            f"{r['wall_seconds']:>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Quantum vs classical energy benchmark")
    ap.add_argument("formula", help="e.g. H2, LiH, H2O, NH3, CH4")
    ap.add_argument("--basis", default="sto3g")
    ap.add_argument("--ansatz", default="uccsd", choices=["uccsd", "two_local"])
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    report = benchmark(from_formula(args.formula), basis=args.basis, ansatz=args.ansatz)
    print(format_table(report))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
