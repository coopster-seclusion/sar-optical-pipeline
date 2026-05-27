"""Config and AOI guardrails.

Call validate_config(cfg, root) near the top of every notebook, after setup(),
to catch misconfiguration before any expensive API calls are made.

Hard errors (ValueError)
    - AOI exceeds 500 km²
    - opera.polarizations is set to an invalid combination

Soft warnings (UserWarning)
    - CRS does not look like a UTM zone EPSG code
    - Any epoch date falls outside Aug–Sep
"""

from __future__ import annotations
import re
import warnings
from datetime import date
from pathlib import Path

import geopandas as gpd

from .utils import auto_utm_crs


_MAX_AOI_KM2: float = 500.0
_VALID_POL_SETS: list[frozenset] = [frozenset({"HH", "HV"}), frozenset({"VV", "VH"})]
_ARCTIC_MONTHS: frozenset[int] = frozenset({8, 9})  # Aug–Sep


def validate_config(cfg: dict, root: Path | str) -> None:
    """Run all guardrail checks against a parsed config.yaml.

    Parameters
    ----------
    cfg  : dict returned by pipeline.env.load_config()
    root : repository root — used to resolve relative AOI paths in the config
    """
    root = Path(root)
    _check_aoi_size(root / cfg["aoi"]["change_area"])
    _check_crs(cfg)
    _check_epoch_windows(cfg)
    _check_polarizations(cfg)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_aoi_size(aoi_path: Path) -> None:
    """Raise ValueError if the change-area AOI exceeds 500 km²."""
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    utm_crs = auto_utm_crs(aoi_path)
    area_km2 = gdf.to_crs(utm_crs).union_all().area / 1_000_000

    if area_km2 > _MAX_AOI_KM2:
        raise ValueError(
            f"AOI is {area_km2:.1f} km² — exceeds the {_MAX_AOI_KM2:.0f} km² maximum.\n"
            "Reduce the study-area polygon before proceeding.\n"
            f"  file : {aoi_path}"
        )


def _check_crs(cfg: dict) -> None:
    """Warn if the configured CRS does not look like a UTM zone EPSG code."""
    crs = cfg.get("crs")
    if crs is None:
        return  # auto-derive mode is always fine

    # UTM North: EPSG:32601–32660   UTM South: EPSG:32701–32760
    if not re.match(r"^EPSG:32[67]\d{2}$", str(crs).upper()):
        warnings.warn(
            f"config.yaml crs='{crs}' does not look like a UTM zone EPSG code.\n"
            "Expected EPSG:326xx (northern hemisphere) or EPSG:327xx (southern).\n"
            "Verify the CRS is appropriate for your AOI, or set crs: null to auto-derive.",
            UserWarning,
            stacklevel=2,
        )


def _check_epoch_windows(cfg: dict) -> None:
    """Warn if any epoch date window falls outside Aug–Sep."""
    outside: list[str] = []
    for epoch in cfg.get("epochs", []):
        for field in ("date_start", "date_end"):
            d = date.fromisoformat(str(epoch[field]))
            if d.month not in _ARCTIC_MONTHS:
                outside.append(
                    f"  year={epoch['year']}  {field}={epoch[field]}  (month {d.month})"
                )

    if outside:
        warnings.warn(
            "The following epoch dates fall outside Aug–Sep (months 8–9).\n"
            "The pipeline targets Arctic snow-free conditions (Aug–Sep).\n"
            "If this is intentional for a different phenology, you can ignore this warning.\n"
            + "\n".join(outside),
            UserWarning,
            stacklevel=2,
        )


def _check_polarizations(cfg: dict) -> None:
    """Raise ValueError if opera.polarizations is set to an invalid combination."""
    pols = cfg.get("opera", {}).get("polarizations")
    if pols is None:
        return  # auto-detect mode is always fine

    pol_set = frozenset(p.upper() for p in pols)
    if pol_set not in _VALID_POL_SETS:
        raise ValueError(
            f"config.yaml opera.polarizations={list(pols)!r} is not a valid dual-pol pair.\n"
            "Must be ['HH', 'HV'] (Arctic / Greenland acquisitions)\n"
            "    or ['VV', 'VH'] (mid-latitude acquisitions).\n"
            "Set to null to auto-detect from the first OPERA search result."
        )
