# Pipeline workflow

All site-specific parameters flow from `config.yaml` and `aoi/` into every notebook.
The three data-acquisition notebooks (01–03) can run in any order; 04–07 must run in sequence.

```mermaid
flowchart TD
    cfg(["config.yaml + aoi/\n― read by every notebook ―"])

    gee[/"Google Earth Engine"/]
    asf_o[/"ASF DAAC · OPERA RTC-S1 30 m"/]
    asf_h[/"ASF · HyP3 Sentinel-1 10 m"/]

    nb01["01 · S2 Retrieval\ndata/s2/"]
    nb02["02 · OPERA Retrieval\ndata/opera_rtc/"]
    nb03["03 · HyP3 Retrieval\ndata/hyp3/"]

    nb04["04 · Align + Stack\nprocessed/"]
    nb05["05 · Change Detection\nchange masks · delta stacks"]

    nb06["06 · Outputs\noutputs/stats/"]
    nb07["07 · Figures\noutputs/figures/"]

    cfg --> nb01 & nb02 & nb03
    gee --> nb01
    asf_o --> nb02
    asf_h --> nb03

    nb01 & nb02 & nb03 --> nb04
    nb04 --> nb05
    nb05 --> nb06 & nb07
```
