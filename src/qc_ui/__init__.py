"""A local web UI for the research graph.

Stdlib only, like the rest of qc_research: `http.server` plus one HTML file.
No Flask, no npm, no build step. Run it and it opens in your browser.

Long-running work (ingest, investigate, agent questions) runs on a background
thread behind a job id, so the browser polls instead of hanging on a request
that can legitimately take minutes.
"""

__version__ = "0.1.0"
