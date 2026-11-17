"""Graph helpers for the examples.

The searches themselves are part of the package — they are not ERA5- or seismic-specific, so they
live in :mod:`nofe.search` where any caller (and the tests) can import them. This module only keeps
the historical import path the example scripts use, so `from graph_utils import radius_edges` keeps
working inside ``examples/``.

``examples/`` is not published; see the README.
"""

from __future__ import annotations

from nofe.search import (
    NearestIndex,
    RadiusIndex,
    knn_edges,
    radius_edges,
    raster_radius_edges,
    sorted_pairs,
    tiled_radius_edges,
)

__all__ = [
    "NearestIndex",
    "RadiusIndex",
    "knn_edges",
    "radius_edges",
    "raster_radius_edges",
    "sorted_pairs",
    "tiled_radius_edges",
]
