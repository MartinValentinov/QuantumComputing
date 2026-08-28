"""The HTTP layer: a JSON API over qc_research, plus one static page.

Binds to localhost only. This serves an unauthenticated API onto a local
database -- fine on your own machine, not something to expose to a network.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qc_research import db, dossier, investigate as inv_mod, queries
from qc_research.pipeline import IngestConfig, ingest_disease
from qc_research.sources import opentargets
from qc_ui import pipeline as showcase
from qc_draw import draw as draw_molecule
from qc_ui.jobs import JobRunner

STATIC = Path(__file__).parent / "static"


class API:
    """Request handlers, kept free of HTTP so they stay testable."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.jobs = JobRunner()
        self._local = threading.local()

    def conn(self) -> sqlite3.Connection:
        """One connection per thread -- SQLite objects are not thread-safe."""
        if not hasattr(self._local, "conn"):
            self._local.conn = db.connect(self.db_path)
        return self._local.conn


    def stats(self, _q: dict) -> dict:
        conn = self.conn()
        diseases = [
            dict(r)
            for r in conn.execute(
                "SELECT d.efo_id, d.name, COUNT(dt.ensembl_id) AS n_targets "
                "FROM disease d LEFT JOIN disease_target dt ON dt.efo_id = d.efo_id "
                "GROUP BY d.efo_id ORDER BY d.name"
            )
        ]
        return {"counts": db.stats(conn), "diseases": diseases}

    def search(self, q: dict) -> dict:
        return {"matches": opentargets.search_disease(q.get("q", ""), size=8)}

    def landscape(self, q: dict) -> dict:
        efo_id = self._efo(q)
        return {"targets": queries.target_landscape(self.conn(), efo_id, limit=200)}

    def gaps(self, q: dict) -> dict:
        efo_id = self._efo(q)
        return {
            "targets": queries.opportunity_gaps(
                self.conn(), efo_id, min_score=float(q.get("min_score", 0.3))
            )
        }

    def failures(self, q: dict) -> dict:
        return queries.failure_patterns(self.conn(), self._efo(q))

    def compounds(self, q: dict) -> dict:
        return {
            "compounds": queries.best_compounds(
                self.conn(), q["symbol"], limit=int(q.get("limit", 25))
            )
        }

    def search_diseases(self, q: dict) -> dict:
        """Live Open Targets search, annotated with what the graph already holds."""
        hits = opentargets.search_disease(q.get("q", ""), size=int(q.get("limit", 12)))
        have = {
            r["efo_id"]: r["n"]
            for r in self.conn().execute(
                "SELECT d.efo_id, COUNT(dt.ensembl_id) n FROM disease d "
                "LEFT JOIN disease_target dt ON dt.efo_id = d.efo_id GROUP BY d.efo_id"
            )
        }
        for h in hits:
            h["in_graph"] = h["efo_id"] in have
            h["n_targets"] = have.get(h["efo_id"], 0)
        return {"matches": hits}

    def cures(self, q: dict) -> dict:
        """Drugs with a curated mechanism against any target for this disease.

        These are real approved and clinical-stage medicines, not predictions --
        the mechanism table only covers compounds a curator has annotated.
        """
        efo_id = self._efo(q)
        rows = [
            dict(r)
            for r in self.conn().execute(
                """SELECT c.chembl_id, c.pref_name, c.smiles, c.max_phase,
                          c.molecular_weight, c.alogp, c.ro5_violations,
                          m.action_type, m.mechanism_of_action,
                          t.symbol, t.name AS target_name, dt.score AS assoc,
                          MAX(a.pchembl_value) AS best_pchembl
                   FROM mechanism m
                   JOIN target t   ON t.chembl_id = m.target_chembl_id
                   JOIN disease_target dt ON dt.ensembl_id = t.ensembl_id
                   JOIN compound c ON c.chembl_id = m.chembl_id
                   LEFT JOIN activity a
                          ON a.chembl_id = c.chembl_id
                         AND a.target_chembl_id = m.target_chembl_id
                   WHERE dt.efo_id = ? AND c.smiles IS NOT NULL
                   GROUP BY c.chembl_id, t.symbol
                   ORDER BY c.max_phase DESC NULLS LAST, dt.score DESC""",
                (efo_id,),
            )
        ]
        return {"efo_id": efo_id, "n": len(rows), "cures": rows}

    def presets(self, _q: dict) -> dict:
        """Ready-made samples, each with a thumbnail of its abnormal proteins."""
        from qc_sample.presets import build_all, highlights

        out = []
        for p in build_all(self.conn()):
            out.append({**p, "highlights": highlights(p["panel"], 7)})
        return {"presets": out}

    def simulate_panel(self, q: dict) -> dict:
        from qc_sample import simulate_panel

        seed = int(q["seed"]) if q.get("seed") else None
        return simulate_panel(
            self.conn(), self._efo(q),
            strength=float(q.get("strength", 1.0)), seed=seed,
        )

    def run_sample(self, body: dict) -> dict:
        from qc_sample import parse_panel

        panel = body.get("panel")
        if isinstance(panel, str):
            panel = parse_panel(panel)
        if not panel:
            raise ValueError("no readable protein values in the panel")

        def work(log):
            return showcase.run_from_sample(self.db_path, panel, log=log)

        return self.jobs.start("sample", work).to_dict()

    def depict(self, q: dict) -> dict:
        smiles = q["smiles"]
        try:
            highlight = {
                int(a) for a in (q.get("highlight") or "").split(",") if a.strip().isdigit()
            }
            svg = draw_molecule(
                smiles,
                width=int(q.get("w", 420)),
                height=int(q.get("h", 320)),
                animate=q.get("animate") in ("1", "true", "yes"),
                highlight=highlight or None,
            )
        except Exception as e:
            return {"error": f"cannot draw: {type(e).__name__}: {e}", "svg": None}
        return {"svg": svg, "smiles": smiles}

    def explain(self, body: dict) -> dict:
        return showcase.explain(body["target"], body["candidate"])

    def dossier(self, q: dict) -> dict:
        d = dossier.build(
            self.conn(), q["symbol"], disease=q.get("disease"), skip_literature=True
        )
        return {
            "symbol": d.symbol,
            "ensembl_id": d.ensembl_id,
            "profile": d.profile,
            "focus_disease": d.focus_disease,
            "chemistry": d.chemistry,
            "clinical": d.clinical,
            "assessment": d.assessment,
        }

    def _efo(self, q: dict) -> str:
        if q.get("efo_id"):
            return q["efo_id"]
        row = self.conn().execute(
            "SELECT efo_id FROM disease WHERE name LIKE ?", (f"%{q.get('disease','')}%",)
        ).fetchone()
        if row:
            return row["efo_id"]
        hits = opentargets.search_disease(q.get("disease", ""), size=1)
        if not hits:
            raise ValueError(f"no disease matched {q.get('disease')!r}")
        return hits[0]["efo_id"]


    def cures(self, q: dict) -> dict:
        """Drugs that act on any target linked to this disease.

        The join is the whole point: Open Targets says which targets matter for
        the disease, ChEMBL says which drugs act on those targets and how far
        each one got. Neither source can answer this alone.
        """
        efo_id = self._efo(q)
        rows = [
            dict(r)
            for r in self.conn().execute(
                """SELECT DISTINCT
                        c.chembl_id, c.pref_name, c.smiles, c.max_phase,
                        c.molecular_weight, c.alogp, c.ro5_violations,
                        m.action_type, m.mechanism_of_action,
                        t.symbol, t.ensembl_id, t.name AS target_name,
                        dt.score AS association_score
                   FROM mechanism m
                   JOIN target t         ON t.chembl_id = m.target_chembl_id
                   JOIN disease_target dt ON dt.ensembl_id = t.ensembl_id
                   JOIN compound c       ON c.chembl_id = m.chembl_id
                  WHERE dt.efo_id = ?
                    AND c.smiles IS NOT NULL
                  ORDER BY m.max_phase DESC NULLS LAST, dt.score DESC""",
                (efo_id,),
            )
        ]
        approved = [r for r in rows if (r["max_phase"] or 0) >= 4]
        return {
            "efo_id": efo_id,
            "n_total": len(rows),
            "n_approved": len(approved),
            "cures": rows,
        }

    def molecule(self, q: dict) -> dict:
        """Everything needed to draw a molecule and say why it works."""
        from qc_draw.features import detect, lipinski
        from qc_draw.layout import layout
        from qc_draw.render import to_svg

        smiles = q.get("smiles")
        record = None
        if q.get("chembl_id"):
            record = self.conn().execute(
                "SELECT * FROM compound WHERE chembl_id = ?", (q["chembl_id"],)
            ).fetchone()
            if record is None:
                raise ValueError(f"no compound {q['chembl_id']}")
            smiles = record["smiles"]
        if not smiles:
            raise ValueError("pass chembl_id or smiles")

        mol, pos = layout(smiles)
        features = detect(mol)
        highlight_name = q.get("highlight")
        highlight = set()
        if highlight_name:
            highlight = next(
                (f.atoms for f in features if f.name == highlight_name), set()
            )

        animate = q.get("animate") == "1"
        svg = to_svg(mol, pos, width=int(q.get("w", 460)), height=int(q.get("h", 340)),
                     highlight=highlight, animate=animate)

        mechanisms = []
        if q.get("chembl_id"):
            mechanisms = [
                dict(r)
                for r in self.conn().execute(
                    """SELECT m.action_type, m.mechanism_of_action, m.max_phase,
                              t.symbol, t.ensembl_id, t.name AS target_name,
                              t.function, t.pathways, t.is_essential, t.loeuf_bin
                         FROM mechanism m
                         LEFT JOIN target t ON t.chembl_id = m.target_chembl_id
                        WHERE m.chembl_id = ?""",
                    (q["chembl_id"],),
                )
            ]

        return {
            "chembl_id": q.get("chembl_id"),
            "name": record["pref_name"] if record else None,
            "smiles": smiles,
            "svg": svg,
            "n_atoms": mol.n_atoms,
            "n_bonds": len(mol.bonds),
            "features": [
                {"name": f.name, "kind": f.kind, "why": f.why, "atoms": sorted(f.atoms)}
                for f in features
            ],
            "lipinski": lipinski(mol),
            "mechanisms": mechanisms,
        }

    def predict(self, q: dict) -> dict:
        """Predicted potency from the from-scratch model, with a domain check."""
        from qc_model.predict import available, get

        symbol = (q.get("symbol") or "").upper()
        if not symbol:
            return {"available_models": available()}

        predictor = get(symbol)
        if predictor is None:
            return {
                "error": f"no trained model for {symbol}",
                "available_models": available(),
                "remedy": f"PYTHONPATH=src python -m qc_model {symbol}",
            }

        smiles = q.get("smiles")
        if not smiles:
            raise ValueError("pass smiles")
        result = predictor.predict_smiles(smiles)
        if result is None:
            return {"error": "could not parse that SMILES"}
        return result

    def quantum(self, q: dict) -> dict:
        """Qubit cost for a small molecule -- cheap, and it never runs VQE."""
        try:
            from qc_quantum.geometry import from_formula
            from qc_quantum.hamiltonian import build_hamiltonian
        except ImportError as e:
            return {"error": "quantum dependencies missing", "detail": str(e)}

        formula = q.get("formula", "H2O")
        try:
            mol = from_formula(formula)
            qp = build_hamiltonian(mol, basis=q.get("basis", "sto3g"))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}",
                    "hint": "only small molecules ASE ships geometries for"}

        n_q = qp.n_qubits
        return {
            "formula": formula, "n_atoms": mol.n_atoms, "n_qubits": n_q,
            "n_pauli_terms": qp.n_pauli_terms,
            "hartree_fock_energy": qp.hf_energy,
            "feasibility": (
                "seconds" if n_q <= 4
                else "many minutes to hours" if n_q <= 12
                else "impractical on a simulator"
            ),
            "note": (
                "Measured on this machine: H2O at 10 qubits took 4,044 s for VQE "
                "to reach chemical accuracy. CCSD(T) got a better answer in "
                "under 0.01 s."
            ),
        }

    def pipeline(self, _q: dict) -> dict:
        """What every package in the repo currently holds. Drives the overview."""
        from qc_model.predict import available

        conn = self.conn()
        counts = db.stats(conn)
        top = [
            dict(r)
            for r in conn.execute(
                """SELECT d.name, d.efo_id, COUNT(DISTINCT dt.ensembl_id) AS targets
                     FROM disease d LEFT JOIN disease_target dt ON dt.efo_id = d.efo_id
                 GROUP BY d.efo_id"""
            )
        ]
        approved = conn.execute(
            "SELECT COUNT(DISTINCT chembl_id) n FROM compound WHERE max_phase >= 4"
        ).fetchone()["n"]

        import os

        return {
            "qc_research": {
                "counts": counts, "diseases": top,
                "sources": ["Open Targets", "ChEMBL", "ClinicalTrials.gov", "PubMed"],
            },
            "qc_quantum": {
                "verified": "H2O, 10 qubits, VQE reached chemical accuracy in 4,044 s",
                "classical_comparison": "CCSD(T) beat it in under 0.01 s",
            },
            "qc_model": {
                "models": available(),
                "approved_compounds_in_graph": approved,
            },
            "qc_design": {
                "tools": 10,
                "agent_ready": bool(os.environ.get("ANTHROPIC_API_KEY")),
            },
            "qc_draw": {"renders": "SMILES to SVG, no cheminformatics library"},
        }


    def ingest(self, body: dict) -> dict:
        disease = body["disease"]
        cfg = IngestConfig(
            n_targets=int(body.get("n_targets", 20)),
            n_trials=int(body.get("n_trials", 250)),
            fetch_literature=bool(body.get("literature", False)),
        )

        def work(log):
            conn = db.connect(self.db_path)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                report = ingest_disease(conn, disease, cfg, verbose=True)
            for line in buf.getvalue().splitlines():
                if line.strip():
                    log(line.rstrip())
            return {
                "disease": report.disease,
                "efo_id": report.efo_id,
                "counts": report.counts,
                "targets_resolved": report.targets_resolved,
                "unmapped": report.targets_unmapped,
                "warnings": report.warnings,
            }

        return self.jobs.start("ingest", work).to_dict()

    def investigate(self, body: dict) -> dict:
        disease = body["disease"]
        strategy = body.get("strategy", "balanced")
        top_n = int(body.get("top_n", 5))

        def work(log):
            conn = db.connect(self.db_path)
            log(f"surveying targets for {disease} ({strategy})")
            inv = inv_mod.investigate(
                conn, disease, strategy=strategy, top_n=top_n,
                skip_literature=True, verbose=False,
            )
            log(f"scored {inv.n_surveyed} targets; deep-dived {len(inv.shortlist)}")
            for m in inv.movement:
                log(f"{m['symbol']}: rank {m['from_rank']} -> {m['to_rank']} ({m['reason']})")
            return {
                "disease": inv.disease,
                "strategy": inv.strategy,
                "n_surveyed": inv.n_surveyed,
                "shortlist": [
                    {
                        "rank": i,
                        "symbol": r["symbol"],
                        "name": r["dossier"].profile.get("name"),
                        "score": r["final_score"]["score"],
                        "components": r["final_score"]["components"],
                        "verdict": r["dossier"].assessment.get("verdict"),
                        "strengths": r["dossier"].assessment.get("strengths"),
                        "risks": r["dossier"].assessment.get("risks"),
                    }
                    for i, r in enumerate(inv.shortlist, 1)
                ],
                "movement": inv.movement,
                "next_actions": inv.next_actions,
                "indication_context": inv.indication_context,
            }

        return self.jobs.start("investigate", work).to_dict()

    def ask(self, body: dict) -> dict:
        question = body["question"]

        def work(log):
            from qc_design.agent import ask as agent_ask

            conn = db.connect(self.db_path)
            log("thinking...")
            run = agent_ask(conn, question, verbose=False)
            for t in run.transcript:
                if t.kind == "tool_call":
                    log(f"→ {t.name}({json.dumps(t.payload)[:80]})")
                elif t.kind == "tool_result":
                    log(f"   {'error' if t.is_error else 'ok'} · {t.seconds:.1f}s")
            return {
                "answer": run.answer,
                "tools_used": run.tools_used(),
                "n_tool_calls": run.n_tool_calls,
                "usage": run.usage,
            }

        return self.jobs.start("ask", work).to_dict()

    def showcase(self, body: dict) -> dict:
        """The full pipeline, as a background job with stage-by-stage progress."""
        disease = body["disease"]
        strategy = body.get("strategy", "balanced")
        use_model = bool(body.get("use_model", True))

        def work(log):
            return showcase.run(self.db_path, disease, strategy=strategy,
                                log=log, use_model=use_model)

        return self.jobs.start("showcase", work).to_dict()

    def job(self, q: dict) -> dict:
        j = self.jobs.get(q.get("id", ""))
        if j is None:
            raise ValueError("unknown job id")
        return j.to_dict()


def make_handler(api: API):
    GET_ROUTES = {
        "/api/stats": api.stats,
        "/api/search": api.search,
        "/api/landscape": api.landscape,
        "/api/gaps": api.gaps,
        "/api/failures": api.failures,
        "/api/compounds": api.compounds,
        "/api/dossier": api.dossier,
        "/api/cures": api.cures,
        "/api/molecule": api.molecule,
        "/api/predict": api.predict,
        "/api/quantum": api.quantum,
        "/api/pipeline": api.pipeline,
        "/api/job": api.job,
        "/api/search_diseases": api.search_diseases,
        "/api/cures": api.cures,
        "/api/depict": api.depict,
        "/api/simulate_panel": api.simulate_panel,
        "/api/presets": api.presets,
    }
    POST_ROUTES = {
        "/api/ingest": api.ingest,
        "/api/investigate": api.investigate,
        "/api/ask": api.ask,
        "/api/showcase": api.showcase,
        "/api/explain": api.explain,
        "/api/run_sample": api.run_sample,
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if path in ("/full", "/full/", "/explore.html"):
                return self._send(200, (STATIC / "explore.html").read_bytes(),
                                  "text/html; charset=utf-8")

            handler = GET_ROUTES.get(path)
            if handler is None:
                return self._json(404, {"error": f"no route {path}"})

            query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            try:
                self._json(200, handler(query))
            except KeyError as e:
                self._json(400, {"error": f"missing parameter {e}"})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            handler = POST_ROUTES.get(path)
            if handler is None:
                return self._json(404, {"error": f"no route {path}"})

            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "body must be JSON"})

            try:
                self._json(200, handler(body))
            except KeyError as e:
                self._json(400, {"error": f"missing field {e}"})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(db_path: str, port: int = 8765, open_browser: bool = True) -> None:
    api = API(db_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(api))
    url = f"http://127.0.0.1:{port}"
    print(f"qc_ui serving {db_path} at {url}")
    print("ctrl-c to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="qc_ui", description="local UI for the research graph")
    ap.add_argument("--db", default=str(db.DEFAULT_DB))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    serve(args.db, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
