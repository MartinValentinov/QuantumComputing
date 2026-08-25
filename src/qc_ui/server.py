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

    # ---- synchronous reads ----

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

    # ---- background jobs ----

    def ingest(self, body: dict) -> dict:
        disease = body["disease"]
        cfg = IngestConfig(
            n_targets=int(body.get("n_targets", 20)),
            n_trials=int(body.get("n_trials", 250)),
            fetch_literature=bool(body.get("literature", False)),
        )

        def work(log):
            conn = db.connect(self.db_path)     # own connection on this thread
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
        "/api/job": api.job,
    }
    POST_ROUTES = {
        "/api/ingest": api.ingest,
        "/api/investigate": api.investigate,
        "/api/ask": api.ask,
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # quiet; errors still surface
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
                html = (STATIC / "index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")

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
