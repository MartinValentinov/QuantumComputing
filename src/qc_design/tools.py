"""The tool surface the agent is allowed to touch.

Design rules, learned the hard way from the rest of this repo:

  * **Errors are return values, not exceptions.** A tool that raises kills the
    loop; a tool that returns {"error": ...} lets the model read the problem
    and try something else. Every handler is wrapped.
  * **Results are JSON-serialisable and small.** Anything unbounded gets a
    `limit` with a sane default -- a 4,000-row dump burns context and teaches
    the model nothing.
  * **Descriptions say when NOT to use the tool.** That is what stops a model
    calling `run_investigation` when it wanted `target_dossier`.
  * **No tool writes anything outside the graph.** This agent reads and
    computes; it does not email, publish, or spend money.
"""

from __future__ import annotations

import json
import sqlite3
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from qc_research import db, dossier, investigate as inv_mod, queries
from qc_research.sources import opentargets


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]

    def spec(self) -> dict:
        """The shape the Messages API expects in `tools`."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _obj(props: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_INT = {"type": "integer"}


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

def _missing_quantum_deps(exc: Exception) -> dict:
    """A tool that cannot run says so as data, with the fix attached."""
    return {
        "error": "quantum_dependencies_missing",
        "detail": str(exc),
        "remedy": (
            "pip install 'qiskit>=2.0' qiskit-nature qiskit-algorithms pyscf ase"
        ),
        "note": (
            "the research tools are unaffected -- only quantum chemistry is "
            "unavailable in this environment"
        ),
    }


def _resolve_disease(conn: sqlite3.Connection, name: str) -> tuple[str, str]:
    row = conn.execute(
        "SELECT efo_id, name FROM disease WHERE name LIKE ?", (f"%{name}%",)
    ).fetchone()
    if row:
        return row["efo_id"], row["name"]
    hits = opentargets.search_disease(name, size=1)
    if not hits:
        raise ValueError(f"no disease matched {name!r}")
    return hits[0]["efo_id"], hits[0]["name"]


def build_registry(conn: sqlite3.Connection) -> dict[str, Tool]:
    """Bind every tool to one database connection."""

    def search_disease(query: str, limit: int = 5) -> dict:
        return {"matches": opentargets.search_disease(query, size=limit)}

    def target_landscape(disease: str, limit: int = 20) -> dict:
        efo_id, name = _resolve_disease(conn, disease)
        rows = queries.target_landscape(conn, efo_id, limit=limit)
        if not rows:
            return {
                "disease": name,
                "targets": [],
                "note": (
                    "the graph holds no targets for this disease -- call "
                    "run_investigation, which ingests before it analyses"
                ),
            }
        return {"disease": name, "efo_id": efo_id, "targets": rows}

    def opportunity_gaps(disease: str, min_score: float = 0.3) -> dict:
        efo_id, name = _resolve_disease(conn, disease)
        rows = queries.opportunity_gaps(conn, efo_id, min_score=min_score)
        return {"disease": name, "n_gaps": len(rows), "targets": rows}

    def target_dossier(symbol: str, disease: str | None = None) -> dict:
        d = dossier.build(conn, symbol, disease=disease, skip_literature=True)
        return {
            "symbol": d.symbol,
            "ensembl_id": d.ensembl_id,
            "name": d.profile.get("name"),
            "target_class": d.profile.get("target_class"),
            "tractability": d.profile.get("tractability"),
            "is_essential": d.profile.get("is_essential"),
            "loeuf_bin": d.profile.get("loeuf_bin"),
            "n_associated_diseases": d.profile.get("n_associated_diseases"),
            "safety_liabilities": d.profile.get("safety_liabilities"),
            "chemical_probes": [
                p["id"] for p in d.profile.get("chemical_probes", []) if p.get("high_quality")
            ],
            "focus_disease": d.focus_disease,
            "chemistry": {
                k: v for k, v in d.chemistry.items() if k != "top_compounds"
            },
            "top_compounds": (d.chemistry.get("top_compounds") or [])[:5],
            "clinical": {k: v for k, v in d.clinical.items() if k != "drugs"},
            "clinical_drugs": (d.clinical.get("drugs") or [])[:8],
            "assessment": d.assessment,
        }

    def best_compounds(symbol: str, limit: int = 10) -> dict:
        rows = queries.best_compounds(conn, symbol, limit=limit)
        if not rows:
            return {
                "symbol": symbol,
                "compounds": [],
                "note": "no measured compounds in the graph for this target",
            }
        return {"symbol": symbol, "compounds": rows}

    def trial_failures(disease: str) -> dict:
        efo_id, name = _resolve_disease(conn, disease)
        out = queries.failure_patterns(conn, efo_id)
        out["disease"] = name
        return out

    def run_investigation(
        disease: str, strategy: str = "balanced", top_n: int = 5
    ) -> dict:
        inv = inv_mod.investigate(
            conn, disease, strategy=strategy, top_n=top_n,
            skip_literature=True, verbose=False,
        )
        return {
            "disease": inv.disease,
            "strategy": inv.strategy,
            "n_surveyed": inv.n_surveyed,
            "shortlist": [
                {
                    "rank": i,
                    "symbol": r["symbol"],
                    "score": round(r["final_score"]["score"], 4),
                    "components": {
                        k: round(v, 3)
                        for k, v in r["final_score"]["components"].items()
                    },
                    "verdict": r["dossier"].assessment.get("verdict"),
                    "strengths": r["dossier"].assessment.get("strengths"),
                    "risks": r["dossier"].assessment.get("risks"),
                }
                for i, r in enumerate(inv.shortlist, 1)
            ],
            "rank_movement": inv.movement,
            "indication_context": inv.indication_context,
            "next_actions": inv.next_actions,
        }

    def compute_molecule_energy(
        formula: str, basis: str = "sto3g", ansatz: str = "uccsd"
    ) -> dict:
        """Quantum + classical energies. Heavy: minutes to hours past ~4 atoms."""
        # ase and pyscf are imported lazily deep inside qc_quantum, so the
        # guard has to wrap the call, not just the import.
        try:
            from qc_quantum.benchmark import benchmark
            from qc_quantum.geometry import from_formula

            return benchmark(from_formula(formula), basis=basis, ansatz=ansatz)
        except ImportError as e:
            return _missing_quantum_deps(e)

    def estimate_quantum_cost(formula: str, basis: str = "sto3g") -> dict:
        """Qubit and Pauli-term count without running anything.

        Always call this before compute_molecule_energy. Qubit count is the
        binding constraint, and it tells you whether the calculation is
        minutes or days.
        """
        try:
            from qc_quantum.geometry import from_formula
            from qc_quantum.hamiltonian import build_hamiltonian

            mol = from_formula(formula)
            qp = build_hamiltonian(mol, basis=basis)
        except ImportError as e:
            return _missing_quantum_deps(e)

        n_q = qp.n_qubits
        return {
            "formula": formula,
            "basis": basis,
            "n_atoms": mol.n_atoms,
            "n_qubits": n_q,
            "n_pauli_terms": qp.n_pauli_terms,
            "hartree_fock_energy": qp.hf_energy,
            "feasibility": (
                "fast" if n_q <= 4
                else "slow -- expect many minutes" if n_q <= 12
                else "impractical on a statevector simulator"
            ),
        }

    def graph_stats() -> dict:
        return db.stats(conn)

    specs: list[Tool] = [
        Tool(
            "search_disease",
            "Resolve a free-text disease name to ontology IDs via Open Targets. "
            "Call this FIRST when the user names a disease informally, because "
            "'lung cancer' is not an identifier and several similar entries "
            "usually exist. Returns candidate matches, best first.",
            _obj({"query": _STR, "limit": _INT}, ["query"]),
            search_disease,
        ),
        Tool(
            "target_landscape",
            "Triage table for a disease: every known target with its association "
            "score, how many active compounds exist, best potency, and how many "
            "drugs reached the clinic. Use for a broad first look. Do NOT use "
            "when you already know which target you care about -- use "
            "target_dossier instead.",
            _obj({"disease": _STR, "limit": _INT}, ["disease"]),
            target_landscape,
        ),
        Tool(
            "opportunity_gaps",
            "Targets that are well evidenced AND small-molecule tractable AND "
            "have no clinical drug. The highest-novelty candidates. Use when "
            "asked what is unexploited or where the opportunity is.",
            _obj({"disease": _STR, "min_score": {"type": "number"}}, ["disease"]),
            opportunity_gaps,
        ),
        Tool(
            "target_dossier",
            "Deep dive on ONE target: biology, tractability, safety "
            "(pan-essentiality, LOEUF genetic constraint, recorded liabilities), "
            "chemical matter, and clinical precedent, plus a rule-based "
            "assessment. This is the tool to reach for before recommending any "
            "target. Pass `disease` to also get the association score.",
            _obj({"symbol": _STR, "disease": _STR}, ["symbol"]),
            target_dossier,
        ),
        Tool(
            "best_compounds",
            "Most potent measured binders for a target, with SMILES, pChEMBL, "
            "molecular weight, logP and Rule-of-5 violations. Use when you need "
            "actual chemical starting points rather than summary counts.",
            _obj({"symbol": _STR, "limit": _INT}, ["symbol"]),
            best_compounds,
        ),
        Tool(
            "trial_failures",
            "Why clinical trials stopped in an indication, bucketed by reason "
            "(efficacy, safety, funding, accrual). Read this before proposing a "
            "design -- it says what actually kills programmes here.",
            _obj({"disease": _STR}, ["disease"]),
            trial_failures,
        ),
        Tool(
            "run_investigation",
            "The full three-pass loop: survey every target, deep-dive the top N, "
            "re-rank with safety data, and return a justified shortlist. "
            "Ingests the disease first if the graph lacks it, so this is the "
            "right tool for a cold start. Expensive -- one call per question, "
            "not per target. Strategy: 'novel' favours unexploited targets, "
            "'fast' favours existing chemistry, 'balanced' splits the difference.",
            _obj(
                {
                    "disease": _STR,
                    "strategy": {"type": "string", "enum": ["novel", "fast", "balanced"]},
                    "top_n": _INT,
                },
                ["disease"],
            ),
            run_investigation,
        ),
        Tool(
            "estimate_quantum_cost",
            "Qubit count, Pauli-term count and Hartree-Fock energy for a "
            "molecule, WITHOUT running the expensive calculation. Always call "
            "this before compute_molecule_energy so you know whether the run is "
            "seconds or days.",
            _obj({"formula": _STR, "basis": _STR}, ["formula"]),
            estimate_quantum_cost,
        ),
        Tool(
            "compute_molecule_energy",
            "Run VQE and the classical ladder (Hartree-Fock, MP2, CCSD, "
            "CCSD(T), FCI) on a small molecule and compare. Returns energies in "
            "Hartree, error against an exact reference, and wall time. VERY "
            "expensive beyond a few atoms -- call estimate_quantum_cost first. "
            "Only accepts small molecules ASE ships geometries for (H2, LiH, "
            "H2O, NH3, CH4). It cannot handle drug-sized molecules.",
            _obj({"formula": _STR, "basis": _STR, "ansatz": _STR}, ["formula"]),
            compute_molecule_energy,
        ),
        Tool(
            "graph_stats",
            "Row counts per table in the research graph. Use to check what data "
            "is actually loaded before assuming a query will return anything.",
            _obj({}, []),
            graph_stats,
        ),
    ]
    return {t.name: t for t in specs}


def dispatch(registry: dict[str, Tool], name: str, args: dict) -> tuple[str, bool]:
    """Run a tool. Returns (json_result, is_error).

    Never raises. A traceback returned as data is something the model can act
    on; an exception is something that ends the run.
    """
    tool = registry.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name!r}"}), True

    try:
        result = tool.handler(**args)
        return json.dumps(result, default=str), False
    except Exception as e:
        return (
            json.dumps(
                {
                    "error": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(limit=3),
                }
            ),
            True,
        )
