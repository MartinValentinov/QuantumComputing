"""Hybrid design agent: an LLM that reasons over the research graph and calls
quantum chemistry as a tool.

The division of labour is the whole point:

    the LLM        decides what to look at, in what order, and when to stop
    qc_research    supplies evidence -- it is never asked to guess
    qc_quantum     supplies physics -- energies, not opinions

Nothing here asks a language model for a fact it could look up. The agent
plans and interprets; the tools measure. That boundary is what makes the
output checkable, because every claim traces to a tool result you can re-run.
"""

__version__ = "0.1.0"
