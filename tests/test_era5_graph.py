import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO / "examples" / "era5" / "utilities"))

from era5_graph import Era5Graph  # noqa: E402

DATA = REPO / "examples" / "era5" / "data" / "2018-01-01.nc"


def _generator(radius: float = 0.05, stride: int = 8) -> Era5Graph:
    return Era5Graph(radius=radius, stride=stride, data_dir=DATA.parent)


def _grid(num_lat: int = 12, num_lon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.linspace(-80.0, 80.0, num_lat)
    longitudes = np.linspace(-180.0, 180.0, num_lon, endpoint=False)
    return latitudes, longitudes


def _features(num_lat: int, num_lon: int, width: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(num_lat, num_lon, width)).astype(np.float32)


def test_positions_are_unit_norm_earth_radii() -> None:
    latitudes, longitudes = _grid()
    points = _generator().positions(latitudes, longitudes)
    assert points.shape == (latitudes.size * longitudes.size, 3)
    assert points.dtype == np.float32
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-6)


def test_positions_join_the_antimeridian() -> None:
    points = _generator().positions(np.array([0.0]), np.array([179.75, -179.75]))
    assert np.linalg.norm(points[0] - points[1]) < np.deg2rad(1.0)


def test_constructor_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="radius must be positive"):
        Era5Graph(radius=0.0)
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        Era5Graph(radius=1.0, stride=0)
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        Era5Graph(radius=1.0, stride=True)


def test_in_sample_graph_has_no_masks_and_no_truth() -> None:
    latitudes, longitudes = _grid()
    data = _generator(radius=0.4).build(_features(12, 24), latitudes, longitudes)

    assert isinstance(data, Data)
    assert data.num_nodes == 12 * 24
    assert data.x.shape == (data.num_nodes, 3)
    assert data.edge_attr.shape[1] == 2 * 3 + 1
    assert data.edge_dist.shape[0] == data.edge_index.shape[1]
    for name in ("input_mask", "output_mask", "y", "observed_index", "withheld_index"):
        assert getattr(data, name, None) is None


def test_merge_orders_observed_rows_first_and_keeps_truth_in_y() -> None:
    latitudes, longitudes = _grid()
    features = _features(12, 24)
    query_mask = np.random.default_rng(1).random(12 * 24) < 0.3
    data = _generator(radius=0.4).build(
        features, latitudes, longitudes, query_mask=query_mask, query_features=features
    )
    num_observed = int((~query_mask).sum())

    assert data.num_nodes == 12 * 24
    assert torch.equal(data.observed_index, torch.as_tensor(np.flatnonzero(~query_mask)))
    assert torch.equal(data.withheld_index, torch.as_tensor(np.flatnonzero(query_mask)))
    # Observed block first: its x rows are the original feature rows, unchanged.
    flat = torch.as_tensor(features.reshape(-1, 3))
    assert torch.equal(data.x[:num_observed], flat[data.observed_index])
    # Truth covers both blocks; the withheld rows' x is the substitute, not their truth.
    assert data.y is not None
    assert data.y.shape == data.x.shape
    assert torch.equal(data.x[:num_observed], data.y[:num_observed])
    assert not torch.equal(data.x[num_observed:], data.y[num_observed:])
    assert torch.equal(data.y[num_observed:], flat[data.withheld_index])
    assert int(data.input_mask.sum()) == num_observed
    assert not bool(data.input_mask[num_observed:].any())
    assert bool(data.output_mask.all())


def test_added_locations_receive_only() -> None:
    latitudes, longitudes = _grid()
    features = _features(12, 24)
    query_mask = np.zeros(12 * 24, dtype=bool)
    query_mask[::4] = True
    data = _generator(radius=0.4).build(features, latitudes, longitudes, query_mask=query_mask)
    num_observed = int((~query_mask).sum())

    senders = set(data.edge_index[0].tolist())
    assert senders <= set(range(num_observed))
    assert data.edge_index.shape[1] > 0
    assert (data.edge_index[0] != data.edge_index[1]).all()


def test_radius_controls_the_neighbourhood() -> None:
    latitudes, longitudes = _grid()
    features = _features(12, 24)
    tight = _generator(radius=0.05).build(features, latitudes, longitudes)
    wide = _generator(radius=0.4).build(features, latitudes, longitudes)
    assert tight.edge_index.shape[1] < wide.edge_index.shape[1]


def test_fill_modes_change_the_substitute() -> None:
    latitudes, longitudes = _grid()
    features = _features(12, 24)
    query_mask = np.zeros(12 * 24, dtype=bool)
    query_mask[::5] = True
    num_observed = int((~query_mask).sum())
    generator = _generator(radius=0.4)

    zeros = generator.build(features, latitudes, longitudes, query_mask=query_mask, fill="zeros")
    assert (zeros.x[num_observed:] == 0).all()
    mean = generator.build(features, latitudes, longitudes, query_mask=query_mask, fill="mean")
    assert torch.allclose(mean.x[num_observed:].mean(dim=0), zeros.x[:num_observed].mean(dim=0))
    interpolated = generator.build(features, latitudes, longitudes, query_mask=query_mask)
    assert torch.isfinite(interpolated.x).all()
    assert not torch.equal(interpolated.x[num_observed:], zeros.x[num_observed:])

    with pytest.raises(ValueError, match="fill must be one of"):
        generator.build(features, latitudes, longitudes, query_mask=query_mask, fill="median")


def test_report_prints_the_measured_neighbour_statistics(capsys) -> None:
    latitudes, longitudes = _grid()
    data = _generator(radius=0.4).build(_features(12, 24), latitudes, longitudes)
    printed = capsys.readouterr().out
    counts = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    assert f"{float(counts.float().mean()):.2f}" in printed
    assert f"{int((counts == 0).sum())} (" in printed
    assert "mean neighbours per receiver" in printed
    assert "nodes without a neighbour" in printed


@pytest.mark.skipif(not DATA.exists(), reason="ERA5 input not fetched")
def test_stride_8_neighbourhood_matches_the_documented_statistics(capsys) -> None:
    """The 2.5 degree choice the notebooks use: mean 11.37, 16 isolated, 182,205 edges."""
    generator = Era5Graph(radius=0.04363, stride=8, data_dir=DATA.parent)
    features, latitudes, longitudes = generator.load("2018-01-01")
    query_mask = np.random.default_rng(0).random(features.shape[0] * features.shape[1]) < 0.2
    data = generator.build(features, latitudes, longitudes, query_mask=query_mask)

    counts = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    assert data.num_nodes == 16_020
    assert abs(float(counts.float().mean()) - 11.37) < 0.2
    assert abs(int((counts == 0).sum()) - 16) <= 5
    assert abs(data.num_edges - 182_205) < 2_000
    assert "mean neighbours per receiver" in capsys.readouterr().out
