"""Automated biomedical research: literature and databases -> a queryable graph.

The thesis of this package: most of what looks like "AI research" in drug
discovery is actually disciplined data integration. Get target, compound,
activity, and trial-outcome data into one schema with provenance, and the
interesting questions become answerable by query rather than by guesswork.

No LLM is required for the ingest layer, and it should stay that way --
the agent reasons *over* this graph, it does not hallucinate it.
"""

__version__ = "0.1.0"
