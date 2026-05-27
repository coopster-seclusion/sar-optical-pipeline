"""HyP3 on-demand SAR processing via Alaska Satellite Facility.

Notebook 03 orchestrates these functions (also used by hyp3_local.py):
  1. search_s1_scenes()     — find S1 IW GRD scenes intersecting the AOI
  2. submit_jobs()          — submit RTC-GAMMA jobs to HyP3
  3. watch_jobs()           — poll until all jobs complete
  4. download_and_process() — download ZIPs, extract, composite, dB-convert, clip, save

All site-specific parameters come from config.yaml via pipeline.env.load_config().
"""

from __future__ import annotations
import re
import time
import logging
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import asf_search as asf
import hyp3_sdk

from .utils import clip_to_aoi, to_db, save_raster

log = logging.getLogger(__name__)

_POL_PATTERN = re.compile(r"_(HH|HV|VV|VH)\.tif$")


# ---------------------------------------------------------------------------
# Scene search
# ---------------------------------------------------------------------------

def search_s1_scenes(
    aoi_path: str | Path,
    date_start: str,
    date_end: str,
    orbit: str = "descending",
) -> list:
    """Search ASF DAAC for Sentinel-1 IW GRD_HD scenes intersecting the AOI.

    Parameters
    ----------
    aoi_path   : path to AOI GeoJSON (WGS-84)
    date_start : ISO date string "YYYY-MM-DD"
    date_end   : ISO date string "YYYY-MM-DD"
    orbit      : "ascending" or "descending"

    Returns
    -------
    list of ASF search result objects
    """
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    wkt = gdf.union_all().wkt
    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        beamMode="IW",
        flightDirection=orbit.upper(),
        intersectsWith=wkt,
        start=date_start,
        end=date_end,
    )
    log.info("S1 GRD search %s–%s → %d scene(s)", date_start, date_end, len(results))
    return list(results)


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_jobs(
    scenes: list,
    cfg: dict,
    hyp3_conn: hyp3_sdk.HyP3,
    year: int,
) -> list[hyp3_sdk.Job]:
    """Submit RTC-GAMMA jobs to HyP3 for a list of S1 scenes.

    Skips scenes that already have a completed job with the same granule name
    to allow safe re-runs after session disconnects.

    Returns
    -------
    list of HyP3 Job objects (submitted + pre-existing completed)
    """
    h_cfg = cfg["hyp3"]
    prefix = f"{h_cfg['job_prefix']}-{year}"

    existing = {
        job.job_parameters["granules"][0]: job
        for job in hyp3_conn.find_jobs(name=prefix).jobs
        if job.status_code == "SUCCEEDED"
    }

    jobs: list[hyp3_sdk.Job] = list(existing.values())
    submitted = 0
    for result in scenes:
        granule = result.properties["sceneName"]
        if granule in existing:
            log.debug("Skipping already-succeeded job for %s", granule)
            continue
        job = hyp3_conn.submit_rtc_job(
            granule=granule,
            name=prefix,
            resolution=h_cfg["resolution"],
            radiometry=h_cfg["radiometry"],
            scale=h_cfg["scale"],
            dem_name=h_cfg["dem"],
            speckle_filter=h_cfg["speckle_filter"],
        )
        jobs.append(job)
        submitted += 1

    log.info("Submitted %d new HyP3 jobs (prefix=%s); %d already done", submitted, prefix, len(existing))
    return jobs


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def watch_jobs(
    hyp3_conn: hyp3_sdk.HyP3,
    jobs: list[hyp3_sdk.Job],
    *,
    timeout: int = 7200,
    poll_interval: int = 60,
) -> list[hyp3_sdk.Job]:
    """Poll HyP3 until all jobs reach a terminal state (SUCCEEDED or FAILED).

    Parameters
    ----------
    timeout       : maximum wait time in seconds (default 2 hours)
    poll_interval : seconds between polls (default 60)

    Returns
    -------
    list of completed Job objects (SUCCEEDED only — FAILED jobs are logged as warnings)
    """
    job_ids = [j.job_id for j in jobs if j.status_code not in ("SUCCEEDED", "FAILED")]
    deadline = time.time() + timeout

    while job_ids and time.time() < deadline:
        time.sleep(poll_interval)
        refreshed = [hyp3_conn.get_job_by_id(jid) for jid in job_ids]
        job_ids = [j.job_id for j in refreshed if j.status_code == "RUNNING"]
        done = [j for j in refreshed if j.status_code != "RUNNING"]
        if done:
            log.info("%d job(s) finished; %d still running", len(done), len(job_ids))

    if job_ids:
        raise TimeoutError(f"HyP3 watch timed out after {timeout}s; {len(job_ids)} job(s) still running")

    succeeded = [j for j in jobs if j.status_code == "SUCCEEDED"]
    failed = [j for j in jobs if j.status_code == "FAILED"]
    for j in failed:
        log.warning("HyP3 job FAILED: %s", j.job_id)

    return succeeded


# ---------------------------------------------------------------------------
# Download and processing
# ---------------------------------------------------------------------------

def download_and_process(
    jobs: list[hyp3_sdk.Job],
    aoi_path: str | Path,
    crs: str,
    out_dir: Path,
    year: int,
    polarizations: list[str] | None = None,
) -> dict[str, Path]:
    """Download HyP3 ZIPs, composite scenes, dB-convert, clip, and save.

    Produces one GeoTIFF per polarization: hyp3_{year}_{POL}.tif

    Parameters
    ----------
    jobs          : list of succeeded HyP3 Job objects
    aoi_path      : path to AOI GeoJSON (WGS-84)
    crs           : target projected CRS string
    out_dir       : root data directory; files written to out_dir/hyp3/
    year          : epoch year (used only for output filename)
    polarizations : explicit list e.g. ["HH", "HV"]; None = auto-detect

    Returns
    -------
    dict mapping polarization string → output Path
    """
    out_dir = Path(out_dir) / "hyp3"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_by_pol: dict[str, list[xr.DataArray]] = {}

    with tempfile.TemporaryDirectory(prefix=f"hyp3_{year}_") as tmp:
        tmp_path = Path(tmp)
        for job in jobs:
            zip_path = tmp_path / f"{job.job_id}.zip"
            job.download_files(location=tmp_path)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)

            for tif in tmp_path.rglob("*.tif"):
                m = _POL_PATTERN.search(tif.name)
                if not m:
                    continue
                pol = m.group(1)
                if polarizations and pol not in polarizations:
                    continue
                da = rxr.open_rasterio(tif, masked=True).squeeze("band", drop=True).astype("float32")
                scenes_by_pol.setdefault(pol, []).append(da)

        outputs: dict[str, Path] = {}
        for pol, scenes in scenes_by_pol.items():
            if len(scenes) == 1:
                composite = scenes[0]
            else:
                composite = xr.concat(scenes, dim="scene").median("scene")

            db = to_db(composite)
            clipped = clip_to_aoi(db.rio.reproject(crs), aoi_path, crs)
            out_path = out_dir / f"hyp3_{year}_{pol}.tif"
            save_raster(clipped, out_path)
            outputs[pol] = out_path
            log.info("Saved %s", out_path)

    return outputs


# ---------------------------------------------------------------------------
# HyP3 connection helper
# ---------------------------------------------------------------------------

def connect(username: str, password: str) -> hyp3_sdk.HyP3:
    """Return an authenticated HyP3 SDK connection."""
    return hyp3_sdk.HyP3(username=username, password=password)
