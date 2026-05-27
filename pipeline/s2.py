"""Sentinel-2 retrieval and preprocessing via Google Earth Engine.

Notebook 01 orchestrates these functions:
  1. Authenticate GEE and load AOI
  2. build_s2_composite()  — cloud-masked, index-enriched median per epoch
  3. export_epoch()        — async GEE → Drive export task
  4. verify_export()       — load a saved TIF and preview it

All site-specific parameters come from config.yaml via pipeline.env.load_config().
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ee  # google-earth-engine; only needed at runtime in GEE-enabled env


# ---------------------------------------------------------------------------
# Cloud / shadow masking
# ---------------------------------------------------------------------------

def mask_s2_scl(image: "ee.Image", scl_mask_classes: list[int]) -> "ee.Image":
    """Apply a pixel-quality mask using the Sentinel-2 Scene Classification Layer.

    Pixels whose SCL value is in scl_mask_classes are masked out.
    Typical mask classes: 3=cloud shadow, 8=med cloud, 9=high cloud,
    10=cirrus, 11=snow/ice.
    """
    import ee
    scl = image.select("SCL")
    bad = scl.remap(scl_mask_classes, [1] * len(scl_mask_classes), 0)
    return image.updateMask(bad.Not())


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------

def add_indices(image: "ee.Image") -> "ee.Image":
    """Add NDVI, NDWI, NDBI, and BSI bands to a scaled (0–1) S2 image.

    Requires bands: B2 (blue), B3 (green), B4 (red), B8 (NIR), B11 (SWIR).
    """
    import ee
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    # Bare soil index
    bsi = (
        image.select("B11").add(image.select("B4"))
        .subtract(image.select("B8").add(image.select("B2")))
        .divide(
            image.select("B11").add(image.select("B4"))
            .add(image.select("B8")).add(image.select("B2"))
        )
        .rename("BSI")
    )
    return image.addBands([ndvi, ndwi, ndbi, bsi])


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------

def build_s2_composite(
    aoi: "ee.Geometry",
    date_start: str,
    date_end: str,
    cfg: dict,
) -> "ee.Image":
    """Build a cloud-masked, median Sentinel-2 composite for one epoch.

    Parameters
    ----------
    aoi        : GEE geometry for the study area (any CRS — GEE handles reprojection)
    date_start : ISO date string, e.g. "2016-08-01"
    date_end   : ISO date string, e.g. "2016-09-30"
    cfg        : parsed config.yaml dict

    Returns
    -------
    ee.Image with bands [B2, B3, B4, B8, B11, NDVI, NDWI, NDBI, BSI]
    in surface reflectance units (0–1).
    """
    import ee
    s2_cfg = cfg["sentinel2"]
    output_bands = s2_cfg["bands"] + ["NDVI", "NDWI", "NDBI", "BSI"]

    collection = (
        ee.ImageCollection(s2_cfg["collection"])
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s2_cfg["cloud_pct_max"]))
        .map(lambda img: mask_s2_scl(img, s2_cfg["scl_mask_classes"]))
        .map(lambda img: img.multiply(0.0001).copyProperties(img, img.propertyNames()))
        .map(add_indices)
    )

    return collection.select(output_bands).median().clip(aoi)


# ---------------------------------------------------------------------------
# GEE export
# ---------------------------------------------------------------------------

def export_epoch(
    image: "ee.Image",
    year: int,
    aoi: "ee.Geometry",
    crs: str,
    drive_folder: str,
    *,
    scale: int = 10,
) -> "ee.batch.Task":
    """Submit an asynchronous GEE export task for one epoch to Google Drive.

    The exported file is named s2_{year}.tif inside drive_folder.
    Call task.status() to poll progress.
    """
    import ee
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"s2_{year}",
        folder=drive_folder,
        fileNamePrefix=f"s2_{year}",
        region=aoi,
        scale=scale,
        crs=crs,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
        maxPixels=1e10,
    )
    task.start()
    return task


# ---------------------------------------------------------------------------
# Post-export verification
# ---------------------------------------------------------------------------

def verify_export(tif_path: str | Path) -> None:
    """Load a saved S2 GeoTIFF and print a brief summary (shape, CRS, value range)."""
    import rioxarray as rxr
    da = rxr.open_rasterio(tif_path)
    print(f"{Path(tif_path).name}: shape={da.shape}, CRS={da.rio.crs}")
    for i, band in enumerate(da.long_name if hasattr(da, "long_name") else range(da.shape[0])):
        vals = da[i].values
        finite = vals[~__import__("numpy").isnan(vals)]
        if finite.size:
            print(f"  band {i}: min={finite.min():.4f}  max={finite.max():.4f}")
