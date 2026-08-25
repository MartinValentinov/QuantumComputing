"""The hybrid loop: Claude plans, the tools measure.

A manual tool-use loop rather than the SDK's `tool_runner`, for two reasons:
the installed SDK predates that helper, and owning the loop means the
transcript, the tool timings, and the stop conditions are all inspectable --
which matters more here than brevity, because the whole point of this design
is that every claim can be traced back to a tool result.

Parallel tool calls are executed in order and their results returned in a
SINGLE user message. Splitting them across messages silently teaches the model
to stop batching, which makes every subsequent run slower.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from qc_design.tools import Tool, build_registry, dispatch

MODEL = "claude-opus-5"
MAX_TOKENS = 64000          # streaming, so a large ceiling costs nothing

SYSTEM = """\
You are a drug discovery research agent. You have tools that read a curated \
biomedical knowledge graph (Open Targets, ChEMBL, ClinicalTrials.gov, PubMed) \
and tools that run quantum and classical electronic-structure calculations.

How to work:

- Look things up. Never state a number you have not read from a tool result. \
If a tool cannot answer, say so rather than filling the gap from memory.
- Cite the evidence inline: association scores, pChEMBL values, phase, NCT IDs, \
LOEUF bins. A claim without a number attached is an opinion, and should be \
labelled as one.
- Check safety before recommending anything. A high association score on a \
pan-essential gene in the most-constrained LOEUF decile is a toxicity problem, \
not an opportunity. Say that plainly.
- Read the failure data. `trial_failures` tells you what actually kills \
programmes in an indication; design around it rather than optimising potency \
in a vacuum.
- Batch independent tool calls in a single turn instead of going one at a time.
- Call `estimate_quantum_cost` before `compute_molecule_energy`. Quantum \
chemistry here is a fragment-scale accuracy probe, not a design engine: it \
handles molecules of a few atoms, and classical CCSD(T) is faster and more \
accurate on everything currently runnable. Never imply otherwise.
- When a tool returns an error, read it and adapt. Errors carry remedies.

Be direct. Lead with the answer, support it with the evidence, and state your \
uncertainty where it exists."""


@dataclass
class Turn:
    kind: str                     # "text" | "tool_call" | "tool_result" | "thinking"
    name: str = ""
    payload: Any = None
    seconds: float = 0.0
    is_error: bool = False


@dataclass
class Run:
    question: str = ""
    answer: str = ""
    transcript: list[Turn] = field(default_factory=list)
    n_tool_calls: int = 0
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)

    def tools_used(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.transcript:
            if t.kind == "tool_call":
                counts[t.name] = counts.get(t.name, 0) + 1
        return counts


def _request_kwargs(messages: list, tools: list[dict]) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": messages,
        "tools": tools,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def _create(client, messages: list, tools: list[dict]):
    """One API round trip, streamed.

    Streaming matters here: tool-using turns with a large `max_tokens` can
    otherwise trip the SDK's HTTP timeout.
    """
    kwargs = _request_kwargs(messages, tools)
    try:
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()
    except TypeError as e:
        # An older SDK rejects `thinking` / `output_config` as unknown kwargs.
        # Drop them and retry rather than failing the whole run.
        if "thinking" not in str(e) and "output_config" not in str(e):
            raise
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()


def ask(
    conn: sqlite3.Connection,
    question: str,
    max_turns: int = 12,
    verbose: bool = True,
    client: Any = None,
) -> Run:
    """Answer one question, calling tools until the model is done."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "the anthropic SDK is not installed -- pip install anthropic"
        ) from e

    client = client or anthropic.Anthropic()
    registry: dict[str, Tool] = build_registry(conn)
    tool_specs = [t.spec() for t in registry.values()]

    run = Run(question=question)
    messages: list[dict] = [{"role": "user", "content": question}]

    def say(m: str) -> None:
        if verbose:
            print(m, flush=True)

    for turn in range(max_turns):
        response = _create(client, messages, tool_specs)
        run.stop_reason = response.stop_reason or ""

        if response.usage:
            for field_name in ("input_tokens", "output_tokens"):
                run.usage[field_name] = run.usage.get(field_name, 0) + getattr(
                    response.usage, field_name, 0
                )

        # A safety decline is a terminal state, not a tool loop.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            run.answer = (
                "The request was declined by a safety classifier"
                + (f" ({detail.category})" if detail else "")
                + "."
            )
            run.transcript.append(Turn("text", payload=run.answer))
            return run

        for block in response.content:
            if block.type == "text" and block.text.strip():
                run.transcript.append(Turn("text", payload=block.text))
            elif block.type == "thinking":
                run.transcript.append(Turn("thinking", payload=getattr(block, "thinking", "")))

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            run.answer = "\n\n".join(
                b.text for b in response.content if b.type == "text"
            ).strip()
            return run

        # Execute every tool_use block, then return ALL results in one message.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            args = block.input if isinstance(block.input, dict) else json.loads(block.input)
            say(f"  → {block.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            run.transcript.append(Turn("tool_call", name=block.name, payload=args))
            run.n_tool_calls += 1

            t0 = time.perf_counter()
            result, is_error = dispatch(registry, block.name, args)
            elapsed = time.perf_counter() - t0

            say(f"    {'error' if is_error else 'ok'} · {elapsed:.1f}s · {len(result)} chars")
            run.transcript.append(
                Turn("tool_result", name=block.name, payload=result,
                     seconds=elapsed, is_error=is_error)
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    **({"is_error": True} if is_error else {}),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    run.answer = (
        f"Stopped after {max_turns} turns without a final answer. "
        f"Tools used: {run.tools_used()}"
    )
    return run


def dry_run(conn: sqlite3.Connection, plan: list[tuple[str, dict]]) -> Run:
    """Execute a fixed tool sequence with no LLM in the loop.

    Exists so the tool surface can be exercised and timed without an API key --
    and so a failing run can be bisected into "the tools are wrong" versus
    "the model chose badly", which are very different problems.
    """
    registry = build_registry(conn)
    run = Run(question="[dry run]")
    for name, args in plan:
        run.transcript.append(Turn("tool_call", name=name, payload=args))
        run.n_tool_calls += 1
        t0 = time.perf_counter()
        result, is_error = dispatch(registry, name, args)
        run.transcript.append(
            Turn("tool_result", name=name, payload=result,
                 seconds=time.perf_counter() - t0, is_error=is_error)
        )
    run.answer = f"executed {run.n_tool_calls} tools"
    return run
