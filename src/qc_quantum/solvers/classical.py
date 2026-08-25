"""Classical reference energies. These are the numbers to beat.

Spoiler: on anything you can actually run, they win. That is the point --
you cannot claim a quantum result means anything without these next to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qc_quantum.geometry import Molecule


@dataclass
class EnergyResult:
    method: str
    energy: float          # total energy, Hartree
    wall_seconds: float
    converged: bool = True
    detail: dict | None = None


def solve_classical(
    mol: Molecule,
    basis: str = "sto3g",
    methods: tuple[str, ...] = ("hf", "mp2", "ccsd", "ccsd(t)", "fci"),
) -> list[EnergyResult]:
    """Run a ladder of classical methods, cheapest first.

    FCI is exact within the basis set but scales factorially -- it is
    skipped automatically when the problem is too large.
    """
    from pyscf import cc, fci, gto, mp, scf

    m = gto.M(
        atom=mol.to_pyscf(),
        basis=basis,
        charge=mol.charge,
        spin=mol.spin,
        verbose=0,
    )

    results: list[EnergyResult] = []

    t0 = time.perf_counter()
    mf = scf.RHF(m).run() if mol.spin == 0 else scf.ROHF(m).run()
    hf_time = time.perf_counter() - t0

    if "hf" in methods:
        results.append(
            EnergyResult("HF", float(mf.e_tot), hf_time, bool(mf.converged))
        )

    if "mp2" in methods:
        t0 = time.perf_counter()
        mp2 = mp.MP2(mf).run()
        results.append(
            EnergyResult("MP2", float(mp2.e_tot), time.perf_counter() - t0)
        )

    cc_obj = None
    if "ccsd" in methods or "ccsd(t)" in methods:
        t0 = time.perf_counter()
        cc_obj = cc.CCSD(mf).run()
        ccsd_time = time.perf_counter() - t0
        if "ccsd" in methods:
            results.append(
                EnergyResult(
                    "CCSD", float(cc_obj.e_tot), ccsd_time, bool(cc_obj.converged)
                )
            )

    if "ccsd(t)" in methods and cc_obj is not None:
        t0 = time.perf_counter()
        et = cc_obj.ccsd_t()
        results.append(
            EnergyResult(
                "CCSD(T)", float(cc_obj.e_tot + et), time.perf_counter() - t0
            )
        )

    if "fci" in methods:
        # FCI determinant count blows up fast; guard before attempting.
        n_orb = int(m.nao)
        if n_orb <= 12:
            t0 = time.perf_counter()
            e_fci = fci.FCI(mf).kernel()[0]
            results.append(
                EnergyResult("FCI", float(e_fci), time.perf_counter() - t0)
            )
        else:
            results.append(
                EnergyResult(
                    "FCI", float("nan"), 0.0, False,
                    {"skipped": f"{n_orb} orbitals is too many for FCI"},
                )
            )

    return results
