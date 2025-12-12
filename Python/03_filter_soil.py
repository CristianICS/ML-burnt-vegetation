"""
Add NDVI and IL values to digitized soil labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import rasterio  # type: ignore

from pyproj import Transformer
from rasterio.io import MemoryFile  # type: ignore

from utils.il import IL
from utils.tile import Tile
# ------------------------------- Config --------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Output path (parent dir will be created if missing)
PNTS_PATH = ROOT / "data" / "labels" / "digitized_soil_25830.gpkg"

# Harmonized images root
IMAGES_PATH = Path(r"F:\Borini\harmoPAF\HarmoPAF_time_series")

# Divided tiles (created inside 01_download_dem.py)
TILES_PATH = ROOT / "data" / "divided_tiles_by_area.gpkg"
TILES = gpd.read_file(TILES_PATH)

# DEM-derived products used to compute IL
ASPECT_VRT = ROOT / "data" / "predictor_variables" / "dem" / "aspect.vrt"
SLOPE_VRT = ROOT / "data" / "predictor_variables" / "dem" / "slope.vrt"

# ------------------------------ Utilities ------------------------------------

def _ensure_out_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def extract_over_points(img_meta, arr, pnts: gpd.GeoDataFrame):
    """Extract the values over points locations inside an array."""
    # Vectorized transformer (fast C code)
    transformer = Transformer.from_crs(
        pnts.crs,
        img_meta["crs"],
        always_xy=True
    )

    with MemoryFile() as memfile:
        with memfile.open(**img_meta) as ds:
            if arr.ndim == 2:
                ds.write(arr[None, :, :])  # add band axis
            else:
                ds.write(arr)

            xs = pnts.geometry.x.to_numpy()
            ys = pnts.geometry.y.to_numpy()
            # Reproject in vectorized form
            xs_r, ys_r = transformer.transform(xs, ys)
            # Build mask directly from coordinate bounds
            left, bottom, right, top = ds.bounds
            mask = (
                (xs_r >= left) & (xs_r <= right) &
                (ys_r >= bottom) & (ys_r <= top)
            )

            coords = np.column_stack((xs_r[mask], ys_r[mask]))
            vals = ds.sample(coords, indexes=1)
            # vals = _to_flat_array(ds.sample(coords, indexes=1))
            return list(vals), mask

# --------------------------------- Main --------------------------------------

def main() -> None:
    _ensure_out_parent(PNTS_PATH)
    soil_points = gpd.read_file(PNTS_PATH)

    # Iterating only subtiles with points.
    selected_tiles = TILES.sjoin(soil_points.to_crs(TILES.crs), how="right")
    # Check for duplicated points (the same point within several tiles)
    if not selected_tiles.index.is_unique:
        # Move the current index (soil point ID) into a column
        # to avoid duplicate indices using idxmax
        selected_tiles = (selected_tiles
            .reset_index().rename(columns={'index': 'point_id'}))
        # Select the tile with more available years
        duration = selected_tiles["final_year"] - selected_tiles["init_year"]
        selected_tiles["duration"] = duration
        # Remove the ones without valid tile data
        selected_tiles.dropna(subset="duration", inplace=True)
        # For duplicated index groups, keep the row with
        # the maximum duration
        max_duration = selected_tiles.groupby("point_id")["duration"].idxmax()
        selected_tiles = selected_tiles.loc[max_duration].set_index("point_id")
    
    # Select tiles containing points
    unique_pairs = selected_tiles[["name", "subtile_fid"]].drop_duplicates()
    pair_index = pd.MultiIndex.from_frame(unique_pairs)
    # Build a MultiIndex for df rows and check membership
    rindex = pd.MultiIndex.from_arrays([TILES["name"], TILES["subtile_fid"]])
    mask = rindex.isin(pair_index)
    # Use mask as condition:
    filtered_tiles = TILES[mask]

    # Initialize target columns if they do not exist
    if "YEAR" not in soil_points.columns:
        soil_points["YEAR"] = np.nan
    if "NDVI" not in soil_points.columns:
        soil_points["NDVI"] = np.nan
    if "IL" not in soil_points.columns:
        soil_points["IL"] = np.nan

    for _, tile in filtered_tiles.iterrows():

        print(f'Process tile {tile["name"]}, subtile {tile["subtile_fid"]}')

        tile_obj = Tile(IMAGES_PATH / tile["name"])
        tile_years = set(tile_obj.get_years())

        # Select the first available year between 2006 and 2023
        year = next((y for y in tile_years if 2006 <= y <= 2023), None)
        # NOTE: The manually digitized soil dataset was selected based on PNOA 
        # ortho imagery, taking into account that each label is soil from 2006 
        # to 2023.
        if year is None:
            print("Skipping...")
            continue
        # Composite & NDVI (summer window to reduce shadows)
        start, end = f"{year}-06-01", f"{year}-07-31"
        tile_obj.filter_date(start, end)
        stack = tile_obj.read_xarr()
        reduced = tile_obj.reduce_xarr(
            stack,
            time_range=[start, end],
            reduce="median",   # "mean", "median", etc.
            gdf=gpd.GeoDataFrame([tile], geometry="geometry", crs=TILES.crs)
        )
        # Compute NDVI
        ndvi = tile_obj.compute_ndvi(reduced).expand_dims({"band": [0]})
        values, valid_mask = tile_obj.xarr_subtract(ndvi, soil_points)

        soil_points.loc[valid_mask, "NDVI"] = values
        soil_points.loc[valid_mask, "YEAR"] = int(year)
        # IL computation and filtering
        il = IL(
            tile_obj.composite_props,
            tile.geometry.bounds,
            TILES.crs
        )

        il_array = il.compute(ASPECT_VRT, SLOPE_VRT)

        il_meta = tile_obj.img_meta.copy()
        il_meta.update({"count": 1, "dtype": il_array.dtype})
        
        values, valid_mask = extract_over_points(
            il_meta,
            il_array,
            soil_points
        )
 
        soil_points.loc[valid_mask, "IL"] = values
        soil_points.loc[valid_mask, "YEAR"] = int(year)

        # Update the selected points adding the new values
        soil_points.to_file(
            PNTS_PATH, layer="suelo_pedro_martin", mode="w", index=False)

if __name__ == "__main__":
    main()
