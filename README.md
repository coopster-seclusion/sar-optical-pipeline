# sar-optical-pipeline

A generalized SAR–optical change detection pipeline for Arctic and sub-Arctic environments. Point it at any area of interest by editing `config.yaml` — no notebook code changes required.

**What it does:**
- Downloads Sentinel-2 SR composites via Google Earth Engine
- Downloads OPERA RTC-S1 (30 m) SAR backscatter via ASF DAAC
- Downloads HyP3 RTC-S1 (10 m) SAR backscatter via Alaska Satellite Facility
- Builds multi-temporal xarray stacks and computes spectral indices
- Produces per-epoch change masks and publication-ready figures

**Original application:** Qaqortoq Airport, Greenland (2016–2025, 7 epochs)

---

## Quick start

### 1. Clone and set up the environment

```
git clone https://github.com/coopster-seclusion/sar-optical-pipeline
cd sar-optical-pipeline
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### 2. Add your credentials

```
cp .env.example .env
```

Open `.env` and fill in:
- `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` — from [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)
- `GEE_PROJECT` — your Google Earth Engine cloud project ID

Authenticate GEE once (run in a notebook or terminal):

```python
import ee
ee.Authenticate()
ee.Initialize(project="your-gee-project-id")
```

### 3. Define your AOI

Place GeoJSON files (WGS-84 / EPSG:4326) in the `aoi/` directory:

| File | Purpose |
|---|---|
| `change_area.geojson` | Polygon(s) where change is expected |
| `stable_reference.geojson` | Nearby stable surface for noise estimation |
| `transect.geojson` | Optional line geometry for backscatter transect profiles |

Example files (generic Svalbard polygons) are included. Replace them with your own.
See `aoi/README.md` for details.

### 4. Edit `config.yaml`

Update the AOI paths, epoch date windows, OPERA burst ID, and CRS.
All site-specific values live in `config.yaml`. See the [Config reference](#config-reference) below.

### 5. Run the notebooks in order

Open in Google Colab or VS Code. Each notebook calls `setup(colab=True/False)` to load
the config and resolve paths automatically.

| Notebook | What it does |
|---|---|
| `01_s2_retrieval.ipynb` | Export Sentinel-2 composites to Google Drive |
| `02_opera_retrieval.ipynb` | Download and preprocess OPERA RTC-S1 tiles |
| `03_hyp3_retrieval.ipynb` | Submit, monitor, and download HyP3 RTC-S1 jobs |
| `04_align_stack.ipynb` | Align sensors, build NetCDF stacks |
| `05_change_detection.ipynb` | Compute deltas, sigma thresholds, change masks |
| `06_outputs.ipynb` | Export statistics and zonal summaries |
| `07_figures.ipynb` | Generate publication-ready figures |

---

## How to point the pipeline at a new AOI

1. **Draw your study area** in [QGIS](https://qgis.org) or [geojson.io](https://geojson.io),
   export as GeoJSON (WGS-84). Keep it under **500 km²**.

2. **Save the files** to `aoi/change_area.geojson` and `aoi/stable_reference.geojson`
   (and optionally `aoi/transect.geojson`).

3. **Edit `config.yaml`** — the only file you need to change:
   - `aoi.change_area` / `aoi.stable_reference` — your file names
   - `epochs` — adjust years and Aug–Sep date windows
   - `opera.burst` — find your burst ID at [search.asf.alaska.edu](https://search.asf.alaska.edu)
     (Product Type: OPERA-S1, draw your AOI, note the burst ID from results)
   - `opera.orbit` — `ascending` or `descending` (check ASF for your site)
   - `crs` — UTM EPSG code for your region, or `null` to auto-derive
   - `hyp3.job_prefix` — short name for your site (used to label HyP3 jobs)
   - `change_detection.baseline_year` — earliest epoch year

4. **Run the notebooks in order.** Guardrails in `pipeline/validate.py` will catch
   common config errors before any API calls are made.

---

## Config reference

| Key | Type | Description |
|---|---|---|
| `aoi.change_area` | path | GeoJSON polygon where change is expected |
| `aoi.stable_reference` | path | GeoJSON polygon over stable surface |
| `aoi.transect` | path or null | Optional line GeoJSON for transect profiles |
| `crs` | EPSG string or null | Output CRS; `null` = auto-derive UTM from AOI centroid |
| `epochs[].year` | int | Calendar year of the monitoring epoch |
| `epochs[].date_start` | date string | Start of imagery search window (`YYYY-MM-DD`) |
| `epochs[].date_end` | date string | End of imagery search window (`YYYY-MM-DD`) |
| `opera.burst` | string | OPERA burst ID, e.g. `T010_020043_IW3` |
| `opera.orbit` | string | `ascending` or `descending` |
| `opera.polarizations` | list or null | `[HH, HV]` or `[VV, VH]`; `null` = auto-detect |
| `sentinel2.cloud_pct_max` | int | Max scene-level cloud % for S2 filter |
| `sentinel2.scl_mask_classes` | list | SCL pixel classes to mask (cloud, shadow, snow) |
| `hyp3.job_prefix` | string | Prefix for HyP3 job names — use your site name |
| `hyp3.resolution` | int | Output pixel spacing in metres (`10` or `30`) |
| `change_detection.baseline_year` | int | Reference year; must appear in `epochs` |
| `change_detection.sigma_threshold_multiplier` | float | Change flagged where \|delta\| > N × sigma |
| `paths.data_dir` | path | Local data directory (relative to repo root) |
| `paths.outputs_dir` | path | Local output directory (relative to repo root) |

---

## Guardrails

`pipeline/validate.py` is called at the top of each notebook and enforces:

| Check | Severity | Rule |
|---|---|---|
| AOI size | **Error** | `change_area` must be ≤ 500 km² |
| Polarization | **Error** | Must be `[HH, HV]` or `[VV, VH]` when explicitly set |
| CRS | Warning | Should be a UTM zone EPSG code (EPSG:326xx or EPSG:327xx) |
| Epoch windows | Warning | Dates should fall within Aug–Sep for Arctic applications |

---

## Data requirements

| Credential | Where to get it |
|---|---|
| NASA Earthdata account | [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) — free |
| Google Earth Engine access | [earthengine.google.com](https://earthengine.google.com) — free for research |
| Google Drive | For GEE export; ~5–50 GB depending on AOI size and epoch count |

---

## License

MIT
