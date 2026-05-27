"""Shared raster utilities used across all pipeline stages."""

from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — registers .rio accessor on DataArray/Dataset
import geopandas as gpd
from pyproj import CRS


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def auto_utm_crs(aoi_path: str | Path) -> str:
    """Derive the UTM CRS string for an AOI GeoJSON from its centroid.

    Returns an EPSG string such as "EPSG:32622".
    """
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    centroid = gdf.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": south, "datum": "WGS84"})
    epsg = crs.to_epsg()
    if epsg is None:
        raise ValueError(f"Could not resolve EPSG for UTM zone {zone} ({'S' if south else 'N'})")
    return f"EPSG:{epsg}"


def resolve_crs(cfg: dict, aoi_path: str | Path) -> str:
    """Return CRS from config, or auto-derive from the AOI centroid if config is null."""
    crs = cfg.get("crs")
    if crs:
        return str(crs)
    return auto_utm_crs(aoi_path)


# ---------------------------------------------------------------------------
# Raster clipping
# ---------------------------------------------------------------------------

def clip_to_aoi(
    da: xr.DataArray,
    aoi_path: str | Path,
    crs: str,
    *,
    all_touched: bool = False,
) -> xr.DataArray:
    """Clip a DataArray to an AOI polygon, returning NaN outside the mask.

    The AOI is reprojected to the raster CRS before clipping.
    """
    gdf = gpd.read_file(aoi_path).to_crs(crs)
    return da.rio.clip(
        gdf.geometry,
        crs=crs,
        drop=True,
        all_touched=all_touched,
        nodata=float("nan"),
    )


# ---------------------------------------------------------------------------
# Backscatter conversion
# ---------------------------------------------------------------------------

def to_db(da: xr.DataArray, *, clip_min: float = -50.0) -> xr.DataArray:
    """Convert linear-power backscatter to decibels (10 × log10).

    Pixels <= 0 become NaN. Result is clipped at clip_min dB to suppress
    the noise floor (default −50 dB).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        db = xr.where(da > 0, 10.0 * np.log10(da.where(da > 0)), float("nan"))
    return db.where(db >= clip_min)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_raster(
    da: xr.DataArray,
    path: str | Path,
    *,
    dtype: str = "float32",
    nodata: float = float("nan"),
) -> None:
    """Write a DataArray to a GeoTIFF (float32, NaN nodata, LZW-compressed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (
        da.astype(dtype)
        .rio.write_nodata(nodata)
        .rio.to_raster(str(path), compress="lzw")
    )


def save_netcdf(
    data: xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    complevel: int = 4,
) -> None:
    """Write a DataArray or Dataset to NetCDF with zlib compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict = {}
    variables = data.data_vars if isinstance(data, xr.Dataset) else {data.name or "data": data}
    for var in variables:
        encoding[var] = {"zlib": True, "complevel": complevel, "dtype": "float32"}
    data.to_netcdf(str(path), encoding=encoding)


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (including parents) for all given paths."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def epoch_years(cfg: dict) -> list[int]:
    """Return the list of epoch years from config, in order."""
    return [e["year"] for e in cfg["epochs"]]


def epoch_by_year(cfg: dict, year: int) -> dict:
    """Return the epoch dict for a given year, raising KeyError if not found."""
    for ep in cfg["epochs"]:
        if ep["year"] == year:
            return ep
    raise KeyError(f"Year {year} not found in config epochs")
