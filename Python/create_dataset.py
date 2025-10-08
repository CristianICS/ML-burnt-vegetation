"""
Join labels from multiple sources into a single dataset.

Sources:
1) Manually digitized labels
2) NFI/IFN (National Forest Inventory)
3) LUCAS
4) SIOSE (bare soil / ground points)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any

import geopandas as gpd
import pandas as pd
import numpy as np

from utils_dataset import Landsat, extract_global

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base data folder: <repo_root>/data
# NOTE: Path(__file__).parents[1] -> parent of the folder containing this script's folder
ROOT = Path(Path(__file__).parents[1], "data")

# Where to save the desired dataset
dataset_output = Path(ROOT.parent, "results", "dataset.gpkg")

# Where to save the Landsat predictor variables
out_folder = Path(ROOT.parent, "results", "tile_extracted_values")
out_folder.mkdir(parents=True, exist_ok=True)  # ensure output folder exists

# Landsat/Sentinel composites images
# TIP: Make this configurable (e.g., via env var) rather than a hard-coded Windows path
images_path = Path(ROOT, "HarmoPAF_time_series")

# Geometry of image tiles (used to filter points within tiles)
tile_bboxes_path = Path(ROOT, "tiles_perimeters.gpkg")

# Target columns for the unified dataset.
# NOTE: "YEAR" was listed but not provided by some sources -> we ensure it exists.
target_cols = ["ESPE", "ESPE_rc", "YEAR", "source", "FCC", "Ocu1", "geometry"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_columns(
    gdf: gpd.GeoDataFrame,
    required_cols: Iterable[str],
    defaults: Dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """
    Ensure the GeoDataFrame contains all required columns.
    Missing columns are added using defaults (or NaN if not provided).
    """
    gdf = gdf.copy()
    defaults = defaults or {}
    for col in required_cols:
        if col not in gdf.columns:
            # Use the provided default or NaN
            gdf[col] = defaults.get(col, np.nan)
    # Reorder columns consistently (geometry is handled by GeoPandas)
    return gdf[required_cols]


def align_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """
    Reproject to target CRS if needed. If gdf has no CRS, assume target CRS.
    """
    gdf = gdf.copy()
    if gdf.crs is None:
        # Warning: in absence of CRS, we *assign* (do not transform).
        # This assumes the coordinates already are in target_crs.
        gdf.set_crs(target_crs, inplace=True)
        return gdf
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


# -----------------------------------------------------------------------------
# Load & unify labels
# -----------------------------------------------------------------------------

labels: list[gpd.GeoDataFrame] = []

# 1) Manually digitized labels (assumed "correct format")
manual = gpd.read_file(Path(ROOT, "labels", "digitized_labels.gpkg"))
# We'll use manual as reference CRS for the rest
ref_crs = manual.crs

# Ensure manual has all target columns (no-ops if already present)
manual = ensure_columns(manual, target_cols)
manual["source"] = manual.get("source", "Digitized")
labels.append(manual)

# 2) Handle NFI labels (IFN2, IFN3, IFN4)
# Include reclassified codes based on mapping file.
codes = pd.read_csv(Path(ROOT, "labels", "label_codes.csv"))

# Build a mapping ESPE -> reclass to avoid join overhead on every call
# IMPORTANT: Make sure both mapping index and IFN['ESPE'] have consistent dtype.
code_map = (
    codes.set_index("code_v1")["code_v1_reclass"]
    .dropna()
)

def add_espe_reclass(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add 'ESPE_rc' by mapping ESPE to reclassified code.
    """
    gdf = gdf.copy()
    # Ensure consistent dtype for the mapping (match index dtype)
    if not code_map.index.dtype == gdf["ESPE"].dtype:
        try:
            gdf["ESPE"] = gdf["ESPE"].astype(code_map.index.dtype)
        except Exception:
            # If coercion fails (e.g., strings mixed with ints), try a safe cast
            gdf["ESPE"] = pd.to_numeric(gdf["ESPE"], errors="coerce")
    gdf["ESPE_rc"] = gdf["ESPE"].map(code_map)
    return gdf

for i in (2, 3, 4):
    ifn = gpd.read_file(Path(ROOT, "labels", f"ifn{i}_labels.gpkg"))
    ifn = add_espe_reclass(ifn)
    if "source" not in ifn.columns:
        ifn["source"] = f"NFI{i}"
    # Ensure required columns and order
    ifn = ensure_columns(ifn, target_cols)
    # Align CRS to manual
    ifn = align_crs(ifn, ref_crs)
    labels.append(ifn)

# 3) Handle SIOSE soil labels (sparse vegetation class)
bare_soil = gpd.read_file(Path(ROOT, "labels", "ground_points_db.gpkg"))

# Broadcast scalar values across the specified columns
bare_soil[["ESPE", "ESPE_rc"]] = 20       # "Sparse vegetation" class
bare_soil[["Ocu1", "FCC"]] = 0            # default zeros as in original code
bare_soil["source"] = "SIOSE"
# YEAR was missing in the original code -> set to NaN (unknown)
bare_soil["YEAR"] = np.nan

bare_soil = ensure_columns(bare_soil, target_cols)
bare_soil = align_crs(bare_soil, ref_crs)
labels.append(bare_soil)

# 4) LUCAS: add soil data from LUCAS (Land Use/Land Cover Area Frame Survey)
# Original path used a raw string with backslashes; prefer Path for portability.
lucas_csv = Path(ROOT, "lucas", "LUCAS_2018_Copernicus_attributes.csv")

# TIP: use 'usecols' to reduce memory footprint if this CSV is large
lucas = pd.read_csv(lucas_csv, low_memory=False)

# Convert to GeoDataFrame using WGS84
# NOTE: 'GPS_LONG' & 'GPS_LAT' are column names in LUCAS 2018 Copernicus attributes.
x_col = lucas["GPS_LONG"]
y_col = lucas["GPS_LAT"]
g = gpd.points_from_xy(x_col, y_col, crs="EPSG:4326")
lucas = gpd.GeoDataFrame(lucas, geometry=g)

# Get points over Aragón (NUTS2 == ES24)
lucas_aragon = lucas[lucas["NUTS2"] == "ES24"].copy()

# Select only "Other bare soil" class (CPRN_LC == 'F4')
lucas_aragon = lucas_aragon.query("CPRN_LC == 'F4'").copy()

# Set target attributes
lucas_aragon[["ESPE", "ESPE_rc"]] = 20
lucas_aragon[["Ocu1", "FCC"]] = 0
lucas_aragon["source"] = "LUCAS"
lucas_aragon["YEAR"] = 2018  # <- important: this was missing

# Ensure required columns and CRS
lucas_aragon = ensure_columns(lucas_aragon, target_cols)
lucas_aragon = lucas_aragon.to_crs(ref_crs)
labels.append(lucas_aragon)

# -----------------------------------------------------------------------------
# Concatenate all sources
# -----------------------------------------------------------------------------

# Once the dataset is created, add predictor variables information
dataset = pd.concat(labels, ignore_index=True)

# Defensive check: drop rows with invalid geometries
# (sometimes joins or mappings can leave NaNs or bad points)
if isinstance(dataset, gpd.GeoDataFrame):
    dataset = dataset[dataset.geometry.notna()].copy()
    dataset = dataset.set_geometry("geometry")
else:
    # Ensure result is GeoDataFrame (pd.concat can downcast if some inputs were DataFrame)
    dataset = gpd.GeoDataFrame(dataset, geometry="geometry", crs=ref_crs)

# -----------------------------------------------------------------------------
# Landsat predictor variable extraction
# -----------------------------------------------------------------------------

tile_bboxes = gpd.read_file(tile_bboxes_path)
# Align tile geometries to dataset CRS (important for spatial filtering)
tile_bboxes = align_crs(tile_bboxes, dataset.crs)

landsat = Landsat(dataset, tile_bboxes, out_folder, images_path)
landsat.batch_extraction()
dataset = landsat.merge_data()

# -----------------------------------------------------------------------------
# Add global predictors
# -----------------------------------------------------------------------------

dataset = extract_global(
    dataset,
    Path(ROOT, "predictor_variables", "dem"),
    Path(ROOT, "predictor_variables", "geologia"),
)

# -----------------------------------------------------------------------------
# Save final dataset
# -----------------------------------------------------------------------------

dataset_output.parent.mkdir(parents=True, exist_ok=True)

# If you prefer not to overwrite, keep the conditional; otherwise, always write.
# Here we overwrite for reproducibility and to avoid stale outputs.
dataset.to_file(dataset_output, index=False, driver="GPKG")
print(f"Saved dataset to: {dataset_output}")
