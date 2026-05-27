"""Change detection — deltas, sigma thresholds, masks, and zonal statistics.

Notebook 05 orchestrates these functions:
  1. compute_delta()           — subtract baseline epoch from each subsequent epoch
  2. sigma_from_baseline()     — noise estimate from stable reference in the baseline year
  3. sigma_temporal()          — noise estimate pooled across all epochs in stable reference
  4. make_changemask()         — binary mask where |delta| exceeds N × sigma
  5. zonal_stats_timeseries()  — per-epoch mean / std / count within AOI polygons
  6. save_delta_stack()        — write delta NetCDF to processed/
  7. save_changemask()         — write binary GeoTIFF per epoch / polarization

All site-specific parameters come from config.yaml via pipeline.env.load_config().
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask

from .utils import save_raster, save_netcdf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_delta(
    stack: xr.DataArray,
    baseline_year: int,
) -> xr.DataArray:
    """Subtract the baseline epoch from every epoch in the stack.

    For SAR stacks the baseline is subtracted directly (dB space → log-ratio).
    For optical index stacks the same arithmetic applies.

    Returns a DataArray with the same dims as stack; the baseline epoch
    has delta = 0 by definition and is retained.
    """
    baseline = stack.sel(time=str(baseline_year), method="nearest")
    delta = stack - baseline
    delta.attrs.update(stack.attrs)
    delta.attrs["baseline_year"] = baseline_year
    return delta


# ---------------------------------------------------------------------------
# Sigma / noise estimation
# ---------------------------------------------------------------------------

def _stable_ref_mask(
    da: xr.DataArray,
    stable_ref_path: str | Path,
    crs: str,
) -> np.ndarray:
    """Return a boolean numpy mask (True = inside stable reference polygon)."""
    gdf = gpd.read_file(stable_ref_path).to_crs(crs)
    transform = from_bounds(
        float(da.x.min()), float(da.y.min()),
        float(da.x.max()), float(da.y.max()),
        da.sizes["x"], da.sizes["y"],
    )
    mask = ~geometry_mask(
        gdf.geometry,
        transform=transform,
        invert=False,
        out_shape=(da.sizes["y"], da.sizes["x"]),
    )
    return mask


def sigma_from_baseline(
    stack: xr.DataArray,
    baseline_year: int,
    stable_ref_path: str | Path,
    crs: str,
) -> float:
    """Estimate noise from the standard deviation of the baseline epoch pixels
    within the stable reference polygon.

    This is Approach (a) from the original pipeline — a single-year,
    spatially-constrained sigma estimate.
    """
    baseline = stack.sel(time=str(baseline_year), method="nearest")
    mask = _stable_ref_mask(baseline, stable_ref_path, crs)
    vals = baseline.values[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("Stable reference contains no finite pixels in baseline epoch")
    return float(np.std(finite))


def sigma_temporal(
    stack: xr.DataArray,
    stable_ref_path: str | Path,
    crs: str,
) -> float:
    """Estimate noise from the temporal standard deviation pooled across all epochs
    within the stable reference polygon.

    This is Approach (b) from the original pipeline — more conservative because
    it includes inter-annual natural variability in the noise estimate.
    """
    sample = stack.isel(time=0)
    mask = _stable_ref_mask(sample, stable_ref_path, crs)
    # Collect all stable-reference pixels across all epochs
    all_vals = stack.values[:, mask]  # shape: (n_epochs, n_stable_pixels)
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        raise ValueError("Stable reference contains no finite pixels across epochs")
    return float(np.std(finite))


# ---------------------------------------------------------------------------
# Change mask
# ---------------------------------------------------------------------------

def make_changemask(
    delta: xr.DataArray,
    sigma: float,
    multiplier: float,
) -> xr.DataArray:
    """Return a binary DataArray: 1 where |delta| > multiplier × sigma, else 0.

    NaN input pixels produce NaN output (not 0) so they are distinguishable
    from "no change detected."
    """
    threshold = multiplier * sigma
    mask = xr.where(np.abs(delta) > threshold, 1.0, 0.0)
    mask = mask.where(np.isfinite(delta))
    mask.attrs["sigma"] = sigma
    mask.attrs["threshold_db"] = threshold
    mask.attrs["multiplier"] = multiplier
    return mask


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def zonal_stats_timeseries(
    stack: xr.DataArray,
    polygons: dict[str, str | Path],
    crs: str,
) -> pd.DataFrame:
    """Compute per-epoch mean, std, and pixel count within named AOI polygons.

    Parameters
    ----------
    stack    : DataArray with dim (time, y, x)
    polygons : dict of {label: geojson_path}, e.g. {"construction": ..., "stable": ...}
    crs      : projected CRS string

    Returns
    -------
    pd.DataFrame with columns: year, aoi, mean, std, count
    """
    rows = []
    for label, geojson_path in polygons.items():
        mask = _stable_ref_mask(stack.isel(time=0), geojson_path, crs)
        for t in stack.time.values:
            da_t = stack.sel(time=t)
            vals = da_t.values[mask]
            finite = vals[np.isfinite(vals)]
            rows.append({
                "year": pd.Timestamp(t).year,
                "aoi": label,
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std":  float(np.std(finite))  if finite.size else float("nan"),
                "count": int(finite.size),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_delta_stack(
    delta: xr.DataArray,
    out_dir: Path,
    name: str,
) -> Path:
    """Write a delta DataArray to out_dir/processed/{name}_delta.nc."""
    path = out_dir / "processed" / f"{name}_delta.nc"
    save_netcdf(delta, path)
    log.info("Saved delta stack → %s", path)
    return path


def save_changemask(
    mask: xr.DataArray,
    out_dir: Path,
    name: str,
    year: int,
) -> Path:
    """Write a binary change mask GeoTIFF to out_dir/figures/{name}_mask_{year}.tif."""
    path = out_dir / "figures" / f"{name}_mask_{year}.tif"
    save_raster(mask, path)
    log.info("Saved change mask → %s", path)
    return path


def save_sigma_report(
    sigmas: dict,
    out_dir: Path,
    filename: str = "05_sigma_thresholds.json",
) -> Path:
    """Write sigma threshold values to a JSON file in out_dir/stats/."""
    path = out_dir / "stats" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(sigmas, fh, indent=2)
    log.info("Saved sigma report → %s", path)
    return path
