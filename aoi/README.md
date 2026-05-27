# AOI Files

Place your study-area GeoJSON files here (WGS-84 / EPSG:4326).

| File | Role |
|---|---|
| `change_area.geojson` | Polygon(s) to monitor for change |
| `stable_reference.geojson` | Nearby stable surface for noise estimation |
| `transect.geojson` | Optional line geometry for backscatter transect profiles |

The `example_*.geojson` files are generic Svalbard polygons for testing.
Copy and rename them, or replace with your own geometries.

Update `config.yaml → aoi:` to point to the correct file names.
