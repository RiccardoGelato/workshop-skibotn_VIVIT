"""The F3 graph builder: lattice offsets, edge assembly and feature scaling.

Every test writes a small synthetic ``seismic.npy``, so the geometry is well defined (the cube's
extent is what normalises the coordinates) and nothing depends on the 1.2 GB benchmark volume.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "seismic" / "utilities"))
from f3_graph import BIN_KM, PADDING_PLANES, F3Cube  # noqa: E402


def _cube(tmp_path, rows: int = 9, cols: int = 13, depth: int = 8, **kwargs) -> F3Cube:
    """An ``F3Cube`` over a synthetic volume of ``rows x cols x depth`` traces."""
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(rows, cols, depth)).astype(np.float32)
    for plane in PADDING_PLANES:
        if plane < depth:
            volume[:, :, plane] = 0.0
    np.save(tmp_path / "seismic.npy", volume)
    return F3Cube(radius_km=kwargs.pop("radius_km", 0.05), data_dir=tmp_path, **kwargs)


def test_offsets_scale_with_the_radius_and_stay_symmetric() -> None:
    # A 50 m radius is two 25 m bins: the 8 neighbours of the first ring plus the 4 axis-aligned
    # ones of the second, which is the 12-edge lattice the demo trains on.
    assert len(F3Cube(radius_km=0.05).offsets()) == 12
    assert len(F3Cube(radius_km=0.025).offsets()) == 4
    # A radius below one bin still connects the four nearest traces rather than nothing.
    assert len(F3Cube(radius_km=0.001).offsets()) == 4
    offsets = F3Cube(radius_km=0.05).offsets()
    assert all((-d_row, -d_col) in offsets for d_row, d_col in offsets)


def test_lattice_edges_give_every_interior_node_the_full_neighbourhood(tmp_path) -> None:
    cube = _cube(tmp_path, rows=21, cols=25)
    features, rows, cols = cube.load()
    data = cube.build(features, rows, cols)
    offsets = len(cube.offsets())

    assert data.num_nodes == rows.size * cols.size
    degrees = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    # Only a trace at least two bins from the border can reach every offset, so the interior check
    # uses the 2-step margin the 50 m radius implies.
    interior = degrees.reshape(rows.size, cols.size)[2:-2, 2:-2]
    assert int(interior.min()) == int(interior.max()) == offsets
    # No self loops, and the edge order is the deterministic (receiver, sender) one.
    assert not bool((data.edge_index[0] == data.edge_index[1]).any())
    assert bool((data.edge_index[1].diff() >= 0).all())


def test_graph_matches_a_radius_search_on_the_same_points(tmp_path) -> None:
    """The lattice offsets are an optimisation, not a different neighbourhood: faiss agrees."""
    from graph_utils import radius_edges

    cube = _cube(tmp_path, rows=15, cols=15)
    features, rows, cols = cube.load()
    data = cube.build(features, rows, cols)

    # A 7x7 block whose interior is two bins from its own border: every offset of those interior
    # traces lands inside the block, so the search is not truncated by the block either. The radius
    # is the lattice's own furthest offset, measured in the coordinates the graph uses.
    grid = data.pos.reshape(rows.size, cols.size, 2)[4:11, 4:11]
    points = grid.reshape(-1, 2)
    spacing = float(points[1, 1] - points[0, 1])
    edge_index, _ = radius_edges(points, round(cube.radius_km / BIN_KM) * spacing + 1e-6)
    searched = torch.bincount(edge_index[1], minlength=points.shape[0]).reshape(7, 7)
    lattice = torch.bincount(data.edge_index[1], minlength=data.num_nodes).reshape(
        rows.size, cols.size
    )[4:11, 4:11]
    assert torch.equal(searched[2:-2, 2:-2], lattice[2:-2, 2:-2])
    # And the interior really does see the full offset set, so the comparison is not vacuous.
    assert int(searched[2:-2, 2:-2].min()) == len(cube.offsets())


def test_a_physical_radius_means_the_same_neighbourhood_at_any_stride(tmp_path) -> None:
    """Normalising coordinates by the cube extent, not the sampled span, is what makes this true."""
    whole = _cube(tmp_path, rows=21, cols=25)
    coarse = F3Cube(radius_km=0.05, data_dir=tmp_path, stride=2)

    whole_features, whole_rows, whole_cols = whole.load()
    coarse_features, coarse_rows, coarse_cols = coarse.load()
    dense = whole.build(whole_features, whole_rows, whole_cols)
    sparse = coarse.build(coarse_features, coarse_rows, coarse_cols)

    # Same physical coordinates for the traces both rasters keep, and the same offset count.
    assert dense.pos[0].tolist() == pytest.approx(sparse.pos[0].tolist())
    assert len(whole.offsets()) == len(coarse.offsets())
    for graph, rows, cols in (
        (dense, whole_rows, whole_cols),
        (sparse, coarse_rows, coarse_cols),
    ):
        degrees = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
        interior = degrees.reshape(rows.size, cols.size)[2:-2, 2:-2]
        assert int(interior.min()) == int(interior.max()) == 12


def test_positions_keep_the_metric_ratio_and_features_are_standardised(tmp_path) -> None:
    cube = _cube(tmp_path, rows=40, cols=50)
    features, rows, cols = cube.load()
    data = cube.build(features, rows, cols)

    # Coordinates are km over the cube's longer side: the long axis spans most of [0, 1] and the
    # short axis keeps its true fraction of it.
    source_rows, source_cols, _ = cube.geometry()
    span = max(source_rows, source_cols) * BIN_KM
    assert np.isclose(float(data.pos[:, 0].max()), ((source_rows - 1) * BIN_KM) / span, rtol=1e-4)
    assert np.isclose(float(data.pos[:, 1].max()), ((source_cols - 1) * BIN_KM) / span, rtol=1e-4)
    # Standardised per vertical sample, not per trace: the depth profile survives.
    assert np.allclose(data.x.mean(dim=0).numpy(), 0.0, atol=1e-4)
    assert np.allclose(data.x.std(dim=0).numpy(), 1.0, atol=1e-3)


def test_inference_raster_reuses_the_training_scale(tmp_path) -> None:
    cube = _cube(tmp_path)
    features, rows, cols = cube.load()
    training = cube.build(features, rows, cols)
    fitted = cube._scale

    other = F3Cube(radius_km=0.05, data_dir=tmp_path)
    other.adopt_scale(cube)
    scaled = other.normalise(features)
    assert other._scale is fitted
    # Same constants, so the inference raster reproduces the training features exactly.
    assert np.array_equal(scaled, training.x.numpy())
    # A cube with no fitted constants refuses to scale rather than inventing its own.
    with pytest.raises(RuntimeError, match="no normalisation constants"):
        F3Cube(radius_km=0.05, data_dir=tmp_path).apply_scale(features)


def test_padding_planes_are_excluded_from_the_feature_axis(tmp_path) -> None:
    """The loader drops the flagged padding planes, so a trace never encodes them."""
    # An 8-sample axis contains plane 0 of the flagged pair but not plane 254, so exactly one plane
    # is dropped: the count must follow the volume, not the constants.
    volume = np.ones((4, 5, 8), dtype=np.float32)
    volume[:, :, 0] = 0.0
    np.save(tmp_path / "seismic.npy", volume)

    cube = F3Cube(radius_km=0.05, data_dir=tmp_path)
    features, rows, cols = cube.load()
    assert cube.geometry() == (4, 5, 7)
    assert features.shape == (4, 5, 7)
    assert np.all(features == 1.0)
