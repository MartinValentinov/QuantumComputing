"""Blood sample in, candidate target out.

A panel of protein concentrations is matched against disease signatures built
from the research graph, which gives a ranked list of what the sample looks
like and which protein is driving it.

Everything here is simulated. Real proteomics carries batch effects, dynamic
range spanning ten orders of magnitude, and the awkward fact that most disease
proteins are not detectable in blood at all. This models the shape of the
problem, not the assay.
"""

from qc_sample.panel import (
    disease_signatures,
    driving_targets,
    match_diseases,
    parse_panel,
    simulate_panel,
)

__all__ = [
    "simulate_panel", "parse_panel", "disease_signatures",
    "match_diseases", "driving_targets",
]
