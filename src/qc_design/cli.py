"""Command line for the hybrid agent.

    python -m qc_design ask "Which undrugged target in pancreatic cancer is the best bet?"
    python -m qc_design tools
    python -m qc_design dry-run
"""

from __future__ import annotations

import argparse
import json
import sys

from qc_research import db


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qc_design", description=__doc__)
    ap.add_argument("--db", default=str(db.DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="ask the agent a question")
    a.add_argument("question")
    a.add_argument("--max-turns", type=int, default=12)
    a.add_argument("--quiet", action="store_true")
    a.add_argument("--transcript", default=None, help="write the full transcript here")

    sub.add_parser("tools", help="list the tool surface the agent can reach")

    d = sub.add_parser("dry-run", help="exercise the tools with no LLM in the loop")
    d.add_argument("--disease", default="pancreatic ductal adenocarcinoma")

    args = ap.parse_args(argv)
    conn = db.connect(args.db)

    if args.cmd == "tools":
        from qc_design.tools import build_registry

        registry = build_registry(conn)
        print(f"{len(registry)} tools\n")
        for t in registry.values():
            required = t.input_schema.get("required", [])
            params = ", ".join(
                f"{k}{'' if k in required else '?'}"
                for k in t.input_schema.get("properties", {})
            )
            print(f"  {t.name}({params})")
            for line in _wrap(t.description, 72):
                print(f"      {line}")
            print()
        return 0

    if args.cmd == "dry-run":
        from qc_design.agent import dry_run

        plan = [
            ("graph_stats", {}),
            ("search_disease", {"query": args.disease, "limit": 2}),
            ("target_landscape", {"disease": args.disease, "limit": 5}),
            ("opportunity_gaps", {"disease": args.disease}),
            ("trial_failures", {"disease": args.disease}),
            ("best_compounds", {"symbol": "KRAS", "limit": 3}),
            ("estimate_quantum_cost", {"formula": "H2"}),
        ]
        run = dry_run(conn, plan)
        print(f"{'tool':<26}{'status':<9}{'sec':>7}{'bytes':>9}")
        print("-" * 51)
        for t in run.transcript:
            if t.kind != "tool_result":
                continue
            payload = json.loads(t.payload)
            status = (
                "error" if t.is_error
                else "no-deps" if isinstance(payload, dict) and "error" in payload
                else "ok"
            )
            print(f"{t.name:<26}{status:<9}{t.seconds:>7.2f}{len(t.payload):>9}")
        print(f"\n{run.answer}")
        return 0

    # ---- ask ----
    from qc_design.agent import ask

    try:
        run = ask(
            conn,
            args.question,
            max_turns=args.max_turns,
            verbose=not args.quiet,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # auth, network, model errors
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nIf this is an auth error: export ANTHROPIC_API_KEY, or run "
            "`ant auth login`.",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 72)
    print(run.answer)
    print("=" * 72)
    print(
        f"\n{run.n_tool_calls} tool calls · {run.tools_used()} · "
        f"stop_reason={run.stop_reason} · usage={run.usage}"
    )

    if args.transcript:
        from dataclasses import asdict
        from pathlib import Path

        Path(args.transcript).write_text(json.dumps(asdict(run), indent=2, default=str))
        print(f"transcript → {args.transcript}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)[:3]


if __name__ == "__main__":
    raise SystemExit(main())
