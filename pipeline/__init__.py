"""
sar-optical-pipeline
====================
Generalized SAR–optical change detection pipeline.

Modules
-------
env      : runtime detection (local vs. Colab) and path resolution
validate : config and AOI guardrails — call validate_config() in every notebook
utils    : shared raster utilities (clip, dB conversion, save, CRS helpers)
s2       : Sentinel-2 retrieval and index computation via Google Earth Engine
opera    : OPERA RTC-S1 search, download, and preprocessing via ASF
hyp3     : HyP3 on-demand SAR job submission, monitoring, and download
stack    : multi-temporal xarray stack construction and NetCDF export
change   : change detection — deltas, sigma thresholds, masks, zonal stats
"""
