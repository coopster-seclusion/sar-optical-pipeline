"""Multi-temporal xarray stack construction and NetCDF export.

Notebook 04 orchestrates these functions:
  1. build_sar_stack()  — load per-epoch SAR TIFs and concat along a time axis
  2. build_s2_stack()   — load per-epoch S2 TIFs, extract index bands, stack temporally
  3. align_to_grid()    — resample one stack to match another sensor's pixel grid
  4. validate_stack()   — sanity checks before saving
  5. save_stack()       — write DataArray or Dataset to NetCDF with compression

Stack naming convention
-----------------------
  opera_{pol}       : OPERA RTC-S1 at native 30 m  (DataArray, dim: time)
  hyp3_{pol}        : HyP3 RTC-S1 at native 10 m   (DataArray, dim: time)
  s2_indices        : Sentinel-2 NDVI/NDWI/NDBI     (Dataset,   dim: time)
  s2_at_opera       : S2 indices resampled to 30 m  (Dataset,   dim: time)
  s2_at_hyp3        : S2 indices resampled to 10 m  (Dataset,   dim: time)
"""

from __future__ import annotations
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

from .utils import save_netcdf

log = logging.getLogger(__name__)

# S2 band index positions (0-based) within the 9-band GeoTIFF produced by notebook 01
# Order: B2, B3, B4, B8, B11, NDVI, NDWI, NDBI, BSI
_S2_INDEX_BANDS = {"NDVI": 5, "NDWI": 6, "NDBI": 7, "BSI": 8}


# ---------------------------------------------------------------------------
# SAR stacks
# ---------------------------------------------------------------------------

def build_sar_stack(
    data_dir: Path,
    sensor: str,
    pol: str,
    epochs: list[int],
) -> xr.DataArray:
    """Load per-epoch SAR GeoTIFFs and concatenate along a labelled time axis.

    Parameters
    ----------
    data_dir : root data directory (contains opera_rtc/ or hyp3/ subdirectories)
    sensor   : "opera" or "hyp3"
    pol      : polarization string, e.g. "HH" or "VV"
    epochs   : ordered list of epoch years to include

    Returns
    -------
    xr.DataArray with dims (time, y, x), time coordinate = pd.DatetimeIndex (Jan 1)
    """
    subdir = "opera_rtc" if sensor == "opera" else "hyp3"
    layers: list[xr.DataArray] = []
    for year in epochs:
        path = data_dir / subdir / f"{sensor}_{year}_{pol}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Expected {sensor} TIF not found: {path}")
        da = rxr.open_rasterio(path, masked=True).squeeze("band", drop=True).astype("float32")
        layers.append(da)

    stack = xr.concat(layers, dim="time")
    stack["time"] = pd.to_datetime([f"{y}-01-01" for y in epochs])
    stack.attrs["sensor"] = sensor
    stack.attrs["polarization"] = pol
    return stack


# ---------------------------------------------------------------------------
# Sentinel-2 stacks
# ---------------------------------------------------------------------------

def build_s2_stack(
    data_dir: Path,
    epochs: list[int],
    index_bands: dict[str, int] | None = None,
) -> xr.Dataset:
    """Load per-epoch S2 GeoTIFFs, extract spectral index bands, and stack.

    Parameters
    ----------
    data_dir    : root data directory (contains s2/ subdirectory)
    epochs      : ordered list of epoch years
    index_bands : mapping of {band_name: 0-based band index} in the GeoTIFF.
                  Defaults to {"NDVI": 5, "NDWI": 6, "NDBI": 7, "BSI": 8}.

    Returns
    -------
    xr.Dataset with variables NDVI, NDWI, NDBI, BSI and dim (time, y, x)
    """
    if index_bands is None:
        index_bands = _S2_INDEX_BANDS

    layers: dict[str, list[xr.DataArray]] = {name: [] for name in index_bands}
    for year in epochs:
        path = data_dir / "s2" / f"s2_{year}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Expected S2 TIF not found: {path}")
        da = rxr.open_rasterio(path, masked=True).astype("float32")
        for name, idx in index_bands.items():
            layers[name].append(da.isel(band=idx).drop_vars("band"))

    time_coord = pd.to_datetime([f"{y}-01-01" for y in epochs])
    arrays = {
        name: xr.concat(slices, dim="time").assign_coords(time=time_coord)
        for name, slices in layers.items()
    }
    return xr.Dataset(arrays)


# ---------------------------------------------------------------------------
# Cross-sensor alignment
# ---------------------------------------------------------------------------

def align_to_grid(
    source: xr.Dataset | xr.DataArray,
    target: xr.DataArray,
    resampling: str = "bilinear",
) -> xr.Dataset | xr.DataArray:
    """Resample source to the pixel grid of target using rioxarray.

    Useful for building cross-sensor stacks (e.g. S2 → OPERA 30 m grid).
    """
    return source.rio.reproject_match(target, resampling=resampling)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_stack(
    stack: xr.DataArray | xr.Dataset,
    expected_epochs: list[int],
    label: str = "",
) -> None:
    """Assert that a stack has the expected epoch count, CRS, and no all-NaN slices."""
    n = len(stack.time) if hasattr(stack, "time") else -1
    if n != len(expected_epochs):
        raise ValueError(f"{label}: expected {len(expected_epochs)} epochs, got {n}")

    if hasattr(stack, "rio") and stack.rio.crs is None:
        raise ValueError(f"{label}: stack has no CRS")

    if isinstance(stack, xr.Dataset):
        arrays = stack.data_vars.values()
    else:
        arrays = [stack]

    for da in arrays:
        for i, t in enumerate(da.time.values):
            if np.all(np.isnan(da.isel(time=i).values)):
                raise ValueError(f"{label}: epoch {t} is entirely NaN")

    log.info("validate_stack OK — %s  epochs=%s  shape=%s", label, list(expected_epochs), stack.dims)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_stack(
    data: xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    complevel: int = 4,
) -> None:
    """Write a stack to NetCDF with float32 zlib compression."""
    save_netcdf(data, path, complevel=complevel)
    log.info("Saved stack → %s", path)
