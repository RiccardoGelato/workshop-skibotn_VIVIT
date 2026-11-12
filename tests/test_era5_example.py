import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "era5.py"


def _load():
    spec = importlib.util.spec_from_file_location("era5_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_imports_without_netcdf_or_matplotlib() -> None:
    module = _load()
    assert module.VARIABLES[0] == "w"
    assert len(module.VARIABLES) == 11


def test_unit_sphere_is_unit_norm() -> None:
    module = _load()
    points = module.unit_sphere(np.array([30.0, 75.0]), np.array([-12.0, 0.0, 33.0]))
    assert points.shape == (6, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-6)


def test_unit_sphere_joins_the_antimeridian() -> None:
    module = _load()
    # 179.75 E and 179.75 W are one grid step apart on the globe, not a hemisphere.
    points = module.unit_sphere(np.array([0.0]), np.array([179.75, -179.75]))
    assert np.linalg.norm(points[0] - points[1]) < np.deg2rad(1.0)


def test_rgba_colours_stay_in_range_and_keep_holes_transparent() -> None:
    module = _load()
    values = np.array([[-5.0, 0.0, 1.0], [0.0, 2.0, 3.0], [5.0, 4.0, -2.0]])
    rgba = module.rgba_colours(values)
    assert rgba.shape == (3, 4)
    # Channels are centred on 0.5 and scaled so the farthest point lands 0.5 away.
    assert np.allclose(np.abs(rgba[:, :3] - 0.5).max(axis=0), 0.5, atol=1e-6)
    assert np.all(rgba[:, 3] == 1.0)

    holed = values.copy()
    holed[1] = np.nan
    holed_rgba = module.rgba_colours(holed)
    # A non-finite row is a hole to leave blank, not a black cell.
    assert holed_rgba[1, 3] == 0.0
    assert np.all(holed_rgba[[0, 2], 3] == 1.0)


def test_rgba_colours_reuse_a_shared_scale() -> None:
    module = _load()
    first = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    second = np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])
    scale = module.colour_scale(first, 5)
    # A panel with a different value range must stay comparable to the first one.
    assert np.allclose(module.rgba_colours(first, scale), module.rgba_colours(first))
    far = module.rgba_colours(second, scale)
    assert np.allclose(far[:, :3], 1.0)


def test_draw_globe_gives_every_point_its_own_quad() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    ccrs = pytest.importorskip("cartopy.crs")
    module = _load()

    latitudes = np.array([-44.0, 0.0, 44.0])
    longitudes = np.array([-120.0, 0.0, 120.0])
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    channels = np.stack([lat_grid.ravel(), lon_grid.ravel(), lat_grid.ravel() * 0.5], axis=1)

    figure, axis = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    try:
        mesh = module.draw_globe(axis, latitudes, longitudes, channels)
        # One quad per grid point: no repeated seam column, no interpolated overhang.
        assert mesh.get_array().shape == (3, 3, 4)
        # "nearest" puts the quad boundary half a step past the outermost centre on both sides.
        xs = axis.get_xlim()
        assert xs[0] == -180.0 and xs[1] == 180.0, xs
        ys = axis.get_ylim()
        assert ys[0] == -66.0 and ys[1] == 66.0, ys
    finally:
        plt.close(figure)


def test_draw_globe_covers_a_full_circle_without_a_seam_gap() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    ccrs = pytest.importorskip("cartopy.crs")
    module = _load()

    # The ERA5 grid: a full circle of longitudes, 2 deg apart, and no pole rows.
    latitudes = np.arange(-88.0, 89.0, 2.0)
    longitudes = np.arange(-180.0, 180.0, 2.0)
    channels = np.zeros((latitudes.size * longitudes.size, 3))

    figure, axis = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    try:
        mesh = module.draw_globe(axis, latitudes, longitudes, channels)
        coordinates = mesh.get_coordinates()
        # 360 deg of quads, not 358: a hand-built edge array would leave a gap at the seam.
        assert np.isclose(coordinates[0, -1, 0] - coordinates[0, 0, 0], 360.0)
        # And the quads reach the poles, half a step past the outermost latitude centre.
        assert np.isclose(coordinates[0, 0, 1], -89.0)
        assert np.isclose(coordinates[-1, 0, 1], 89.0)
    finally:
        plt.close(figure)


def test_draw_globe_rejects_a_shape_that_is_not_one_row_per_point() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    ccrs = pytest.importorskip("cartopy.crs")
    module = _load()

    figure, axis = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    try:
        with pytest.raises(ValueError, match="one per grid point"):
            module.draw_globe(axis, np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.zeros((3, 3)))
    finally:
        plt.close(figure)
