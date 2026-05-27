"""Runtime environment detection and path resolution.

Notebooks import this module to get consistent paths whether they run
locally or inside Google Colab with a mounted Drive.

Typical usage
-------------
    from pipeline.env import setup
    cfg, DATA, OUT = setup()          # local
    cfg, DATA, OUT = setup(colab=True) # Colab — mounts Drive automatically
"""

from __future__ import annotations
from pathlib import Path
import yaml


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def repo_root() -> Path:
    """Return the repository root directory.

    Locally this is the parent of the pipeline/ package.
    In Colab notebooks clone the repo to /content/sar-optical-pipeline by default.
    """
    if is_colab():
        return Path("/content/sar-optical-pipeline")
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml and return it as a dict.

    Defaults to <repo_root>/config.yaml when config_path is omitted.
    """
    if config_path is None:
        config_path = repo_root() / "config.yaml"
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _drive_root() -> Path:
    """Mount Google Drive (Colab only) and return the MyDrive root."""
    if not is_colab():
        raise RuntimeError("_drive_root() called outside Colab")
    from google.colab import drive  # type: ignore[import]
    drive.mount("/content/drive")
    return Path("/content/drive/MyDrive")


def setup(
    config_path: str | Path | None = None,
    colab: bool | None = None,
    drive_subdir: str = "sar_optical_pipeline",
) -> tuple[dict, Path, Path]:
    """One-call notebook bootstrap.

    Returns
    -------
    cfg : dict
        Parsed config.yaml.
    data_dir : Path
        Directory where sensor GeoTIFFs and NetCDF stacks are stored.
    out_dir : Path
        Directory where figures and CSV exports are written.

    Parameters
    ----------
    config_path :
        Path to config.yaml; defaults to <repo_root>/config.yaml.
    colab :
        Force Colab mode (True) or local mode (False). Defaults to
        auto-detection via is_colab().
    drive_subdir :
        Subdirectory under MyDrive used in Colab mode.
    """
    cfg = load_config(config_path)
    in_colab = is_colab() if colab is None else colab

    if in_colab:
        drive = _drive_root()
        data_dir = drive / drive_subdir / "data"
        out_dir = drive / drive_subdir / "outputs"
    else:
        root = repo_root()
        data_dir = root / cfg["paths"]["data_dir"]
        out_dir = root / cfg["paths"]["outputs_dir"]

    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, data_dir, out_dir
