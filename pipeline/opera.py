"""OPERA RTC-S1 retrieval and preprocessing via ASF DAAC.

Notebook 02 orchestrates these functions:
  1. suggest_burst_ids() — discover valid burst IDs for a new AOI (setup helper)
  2. search_epoch()      — query ASF for OPERA products matching burst / date window
  3. process_epoch()     — download, median-composite, dB-convert, clip, save

All site-specific parameters come from config.yaml via pipeline.env.load_config().
"""

from __future__ import annotations
import re
import time
import tempfile
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import asf_search as asf
import requests

from .utils import clip_to_aoi, to_db, save_raster

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Burst discovery helper
# ---------------------------------------------------------------------------

def suggest_burst_ids(
    aoi_path: str | Path,
    orbit: str | None = None,
    *,
    search_year: int = 2023,
) -> list[str]:
    """Return OPERA burst IDs that intersect the AOI.

    Run this once when setting up a new site to find the correct burst ID
    before populating opera.burst in config.yaml.

    Parameters
    ----------
    aoi_path    : path to change-area GeoJSON (WGS-84)
    orbit       : "ascending" or "descending" to filter; None = return both
    search_year : calendar year used for the discovery search window (Aug–Sep)

    Returns
    -------
    Sorted list of unique burst ID strings, e.g. ["T010_020043_IW3", ...]
    """
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    wkt = gdf.union_all().wkt

    kwargs: dict = dict(
        dataset="OPERA_L2_RTC-S1_V1",
        intersectsWith=wkt,
        start=f"{search_year}-08-01",
        end=f"{search_year}-09-30",
    )
    if orbit:
        kwargs["flightDirection"] = orbit.upper()

    results = asf.search(**kwargs)
    burst_ids = sorted(
        {r.properties.get("operaBurstID", "") for r in results} - {""}
    )

    if not burst_ids:
        print(
            "No OPERA results found intersecting this AOI.\n"
            "Try a different search_year or check that the AOI is in a Sentinel-1 coverage area."
        )
    else:
        print(f"Found {len(burst_ids)} burst ID(s) intersecting your AOI:")
        for bid in burst_ids:
            print(f"  {bid}")
        print("\nCopy the relevant ID into config.yaml → opera.burst")

    return burst_ids


# ---------------------------------------------------------------------------
# ASF search
# ---------------------------------------------------------------------------

def search_epoch(
    cfg: dict,
    date_start: str,
    date_end: str,
) -> list:
    """Search ASF DAAC for OPERA RTC-S1 products covering one epoch.

    Parameters
    ----------
    cfg        : parsed config.yaml dict
    date_start : ISO date string "YYYY-MM-DD"
    date_end   : ISO date string "YYYY-MM-DD"

    Returns
    -------
    list of ASF search result objects for the configured burst / orbit
    """
    opera_cfg = cfg["opera"]
    if not opera_cfg.get("burst"):
        raise ValueError("config.yaml: opera.burst must be set (e.g. 'T010_020043_IW3')")

    results = asf.search(
        dataset=opera_cfg["collection"],
        processingLevel="RTC",
        flightDirection=opera_cfg["orbit"].upper(),
        start=date_start,
        end=date_end,
        operaBurstID=[opera_cfg["burst"]],
    )
    log.info("OPERA search %s–%s → %d result(s)", date_start, date_end, len(results))
    return list(results)


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------

def _backscatter_urls(result, polarizations: list[str]) -> dict[str, str]:
    """Resolve per-polarization backscatter TIF URLs from an ASF result object.

    Handles three OPERA product formats encountered in the wild:
      NRT (2024+)      — url list contains per-pol TIF URLs directly
      Reprocessed backlog — additionalUrls contains the backscatter TIFs
      Fallback         — derive URL from the browse PNG pattern
    """
    props = result.properties
    urls: dict[str, str] = {}

    # 1. NRT format: url is a list; find per-polarization TIF entries
    for url in props.get("url", []):
        for pol in polarizations:
            if re.search(rf"_{pol}\.tif$", url, re.IGNORECASE):
                urls[pol] = url

    if urls:
        return urls

    # 2. Reprocessed format: backscatter TIFs in additionalUrls
    for url in props.get("additionalUrls") or []:
        for pol in polarizations:
            if re.search(rf"_{pol}\.tif$", url, re.IGNORECASE):
                urls[pol] = url

    if urls:
        return urls

    # 3. Last resort: derive from browse URL (confirmed pattern in ASF catalog)
    browse = props.get("browse", "")
    if browse:
        base = re.sub(r"_browse\.png$", "", browse)
        for pol in polarizations:
            urls[pol] = f"{base}_{pol}.tif"

    missing = [p for p in polarizations if p not in urls]
    if missing:
        raise RuntimeError(
            f"Could not resolve URLs for polarizations {missing} "
            f"from {props.get('granuleName', '(unknown)')}"
        )
    return urls


def _resolve_polarizations(cfg: dict, results: list) -> list[str]:
    """Return explicit polarizations from config, or detect from the first result."""
    pols = cfg["opera"].get("polarizations")
    if pols:
        return list(pols)
    if not results:
        raise RuntimeError("No OPERA results to detect polarizations from")
    granule = results[0].properties.get("granuleName", "")
    if "HH" in granule or "HV" in granule:
        return ["HH", "HV"]
    return ["VV", "VH"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _fetch_file(url: str, dest_dir: Path, session: requests.Session, *, retries: int = 3) -> Path:
    """Download a single file to dest_dir, retrying on transient errors."""
    dest = dest_dir / Path(url).name
    if dest.exists():
        return dest
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            return dest
        except Exception as exc:
            if attempt == retries:
                raise
            log.warning("Download attempt %d/%d failed (%s); retrying…", attempt, retries, exc)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


# ---------------------------------------------------------------------------
# Per-epoch processing
# ---------------------------------------------------------------------------

def process_epoch(
    year: int,
    results: list,
    aoi_path: str | Path,
    crs: str,
    out_dir: Path,
    session: requests.Session,
    cfg: dict,
) -> dict[str, Path]:
    """Download, composite, dB-convert, clip, and save OPERA data for one epoch.

    Produces one GeoTIFF per polarization: opera_{year}_{POL}.tif

    Returns
    -------
    dict mapping polarization string → output Path
    """
    polarizations = _resolve_polarizations(cfg, results)
    out_dir = out_dir / "opera_rtc"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix=f"opera_{year}_") as tmp:
        tmp_path = Path(tmp)
        for pol in polarizations:
            scenes: list[xr.DataArray] = []
            for result in results:
                urls = _backscatter_urls(result, polarizations)
                tif = _fetch_file(urls[pol], tmp_path, session)
                da = rxr.open_rasterio(tif, masked=True).squeeze("band", drop=True)
                scenes.append(da.astype("float32"))

            # Median composite in linear power domain → dB → clip to AOI
            if len(scenes) == 1:
                composite = scenes[0]
            else:
                composite = xr.concat(scenes, dim="scene").median("scene")

            db = to_db(composite)
            clipped = clip_to_aoi(db.rio.reproject(crs), aoi_path, crs)

            out_path = out_dir / f"opera_{year}_{pol}.tif"
            save_raster(clipped, out_path)
            outputs[pol] = out_path
            log.info("Saved %s", out_path)

    return outputs


# ---------------------------------------------------------------------------
# Earthdata session helper
# ---------------------------------------------------------------------------

def earthdata_session(username: str, password: str) -> requests.Session:
    """Return a requests.Session pre-authenticated for NASA Earthdata."""
    session = requests.Session()
    session.auth = (username, password)
    session.get("https://urs.earthdata.nasa.gov", timeout=30)
    return session
