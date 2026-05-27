"""OPERA RTC-S1 retrieval and preprocessing via ASF DAAC.

Notebook 02 orchestrates these functions:
  1. search_epoch()        — query ASF for OPERA products matching burst / date window
  2. process_epoch()       — download, median-composite, dB-convert, clip, save

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
# ASF search
# ---------------------------------------------------------------------------

def search_epoch(
    cfg: dict,
    date_start: str,
    date_end: str,
    session: requests.Session,
) -> list:
    """Search ASF DAAC for OPERA RTC-S1 products covering one epoch.

    Parameters
    ----------
    cfg        : parsed config.yaml dict
    date_start : ISO date string "YYYY-MM-DD"
    date_end   : ISO date string "YYYY-MM-DD"
    session    : authenticated requests.Session (Earthdata credentials)

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

    Handles both NRT (direct TIF URLs in properties) and reprocessed
    (HDF5-based) product formats.
    """
    props = result.properties
    urls: dict[str, str] = {}

    # NRT format: urls listed directly in the result URLs list
    for url in props.get("url", []):
        for pol in polarizations:
            if f"_{pol}.tif" in url:
                urls[pol] = url

    # Reprocessed format: construct URL from the HDF5 browse URL pattern
    if not urls:
        browse = props.get("browse", "")
        base = re.sub(r"_browse\.png$", "", browse)
        for pol in polarizations:
            urls[pol] = f"{base}_{pol}.tif"

    missing = [p for p in polarizations if p not in urls]
    if missing:
        raise RuntimeError(f"Could not resolve URLs for polarizations {missing} from {props.get('granuleName')}")
    return urls


def _resolve_polarizations(cfg: dict, results: list) -> list[str]:
    """Return explicit polarizations from config, or detect from the first result."""
    pols = cfg["opera"].get("polarizations")
    if pols:
        return list(pols)
    # Auto-detect: inspect filename of first result for HH/HV or VV/VH
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
    # Follow redirects through the URS OAuth flow
    session.get("https://urs.earthdata.nasa.gov", timeout=30)
    return session
