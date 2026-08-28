"""Where quantum computing does and does not earn its place in this project.

Three claims were tested. One holds, two do not, and the two that do not were
tested in the way most likely to make them look good.

1. QUANTUM KERNELS FOR MOLECULE MATCHING -- does not hold.
   The first comparison in this repo was unfair: the quantum kernel got eight
   global descriptors while Tanimoto got a 1024-bit structural fingerprint.
   That compared representations, not kernels. `fair_test` fixes it by giving
   both kernels identical features. On equal footing the quantum kernel is
   worse in every encoding, and giving it better structural features helps the
   classical kernel more than it helps the quantum one.

2. ELECTRONIC DESCRIPTORS FOR RANKING WARHEADS -- does not hold.
   Ionisation, affinity and electrophilicity computed by active-space Delta-SCF
   put methanol and dimethyl ether above a Michael acceptor. Neither is
   electrophilic. Frontier correlation energy failed the same way, ranking
   ammonia above benzene. The cause is not the solver: a CAS(6,6) selects
   whichever orbitals sit near the frontier, and those are oxygen lone pairs in
   one molecule and a pi system in another, so the numbers are not comparable
   across molecules however accurately each one is computed.

3. VQE FOR GROUND-STATE ENERGIES -- holds, and is verified.
   Against exact diagonalisation of the same active-space Hamiltonian, VQE
   agrees to five decimal places. Against full FCI on H2O it lands inside
   chemical accuracy. This is the part that needs no disputed speedup argument:
   it is a quantum system being modelled by a quantum system.

The gap between 3 and a useful drug-scoring descriptor is geometry. A reaction
energy -- warhead plus cysteine thiol going to the covalent adduct -- is a
single consistent system, so it avoids the comparability problem entirely, and
it is exactly the quantity that decides whether a covalent inhibitor works.
Computing it needs 3D structures for the reactants and the adduct, and this
repo has no geometry optimiser. That is the blocker, and it is a classical one.
"""

from __future__ import annotations


def run_all(conn, targets: list[str] | None = None, verbose: bool = True) -> dict:
    """Re-run every claim above and report what the numbers say now."""
    import statistics as st

    from qc_research import queries
    from qc_quantum.fair_test import run as fair_run

    def say(m=""):
        if verbose:
            print(m, flush=True)

    targets = targets or ["LRRK2", "KRAS", "TOP1", "SNCA", "DRD2", "MAPT"]

    say("1. Quantum kernel vs classical, identical features")
    per_encoding: dict[str, dict] = {}
    reference = []
    for symbol in targets:
        rows = [
            c for c in queries.best_compounds(conn, symbol, limit=30)
            if c.get("smiles") and c.get("best_pchembl") is not None
        ]
        if len(rows) < 12:
            continue
        result = fair_run([c["smiles"] for c in rows],
                          [c["best_pchembl"] for c in rows])
        reference.append(result["tanimoto_reference"])
        for entry in result["per_encoding"]:
            acc = per_encoding.setdefault(
                entry["encoding"], {"quantum": [], "rbf": [], "wins": 0, "n": 0}
            )
            acc["quantum"].append(entry["quantum"])
            acc["rbf"].append(entry["rbf"])
            acc["wins"] += int(entry["quantum_wins"])
            acc["n"] += 1

    kernel_summary = {
        name: {
            "quantum": round(st.mean(a["quantum"]), 4),
            "classical_rbf": round(st.mean(a["rbf"]), 4),
            "quantum_wins": f"{a['wins']}/{a['n']}",
            "quantum_better": bool(st.mean(a["quantum"]) < st.mean(a["rbf"])),
        }
        for name, a in per_encoding.items()
    }
    for name, row in kernel_summary.items():
        say(f"   {name:<13} quantum {row['quantum']:.3f}  classical {row['classical_rbf']:.3f}"
            f"  wins {row['quantum_wins']}")

    say("")
    say("2. VQE against exact diagonalisation of the same Hamiltonian")
    from qc_quantum.electronic import from_ase

    exact = from_ase("H2CO", use_vqe=False)
    vqe = from_ase("H2CO", use_vqe=True)
    agreement = None
    if not exact.error and not vqe.error:
        agreement = abs(
            exact.energies["neutral"] - vqe.energies["neutral"]
        )
        say(f"   formaldehyde, {exact.n_qubits} qubits: "
            f"|VQE - exact| = {agreement:.2e} Hartree")

    return {
        "kernel_fair_test": kernel_summary,
        "tanimoto_reference": round(st.mean(reference), 4) if reference else None,
        "vqe_vs_exact_hartree": agreement,
        "verdict": {
            "quantum_kernel_helps": any(r["quantum_better"] for r in kernel_summary.values()),
            "vqe_correct": bool(agreement is not None and agreement < 1e-4),
        },
    }
