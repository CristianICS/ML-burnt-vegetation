"""
Generate DEM-derived predictor variables per Landsat image tile.

Workflow
--------
1) Load the tiles layer (vector footprints of Landsat image tiles).
2) For each tile, split it into sub-tiles of ~50,000 hectares (~500 km2).
3) For every sub-tile:
   - Download a DEM clipped to the sub-tile extent.
   - Compute DEM-derived products (e.g., slope, aspect, roughness; see utils_dem).
   - Save outputs with a numeric suffix identifying the sub-tile.
4) Export a GeoPackage with all sub-tile polygons, carrying the parent tile attributes.

Assumptions
-----------
- utils_dem.py provides: split_tile_by_area, download_dem, create_divided_tile_gpd, compute_dem_products.
- The tiles layer has a "name" field used as a unique identifier.
- All DEM computations are handled inside compute_dem_products.
"""

from pathlib import Path
from utils_dem import split_tile_by_area
from utils_dem import download_dem
from utils_dem import create_divided_tile_gpd
from utils_dem import compute_dem_products
import geopandas as gpd

# Project root (two levels above this file). Adjust if your layout differs.
ROOT = Path(__file__).resolve().parent.parent

# Output folder for DEM predictor rasters per tile.
output_dir = Path(ROOT, "data/predictor_variables/dem")

# Root folder containing harmonized Landsat scenes and the tiles layer.
images_dir = Path(ROOT, r"HarmoPAF_time_series")

# -----------------------------------------------------------------------------
# 1) Load tile footprints
# -----------------------------------------------------------------------------
# The tiles layer is expected to be a GeoPackage
# and must include a "name" attribute.
layer_path = Path(ROOT, "data", "tiles_perimeters.gpkg")
tiles = gpd.read_file(layer_path)

# This list will collect, for every original tile, the list of Shapely boxes
# representing the sub-tiles created by area-based splitting.
divided_tiles_boxes = []

# -----------------------------------------------------------------------------
# 2) Iterate over tiles and split each into ~50,000 ha sub-tiles
# -----------------------------------------------------------------------------
for i, tile in tiles.iterrows():

    print(f'Computing tile {tile["name"]}')

    # Split the tile polygon into rectangular boxes of ~50,000 hectares
    # Returns:
    #   bboxes - list of bounding boxes in the CRS units (for IO/processing)
    #   bboxes_shapely - list of Shapely polygons for export/attribution
    bboxes, bboxes_shapely = split_tile_by_area(tile, 50000)
    divided_tiles_boxes.append(bboxes_shapely)

    # Destination folder for all outputs derived from this tile.
    i_out_dir = Path(output_dir, tile["name"])

    # If you want to skip reprocessing tiles that already have outputs,
    # uncomment the next two lines:
    # if i_out_dir.exists():
    #     continue

    # Ensure the per-tile output directory exists
    # (uncomment to create on the fly):
    # i_out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3) For each sub-tile, fetch DEM and compute derivative products
    # -------------------------------------------------------------------------
    # Suffix indexes sub-tiles as _1, _2, ... to keep filenames unique
    # and traceable.
    suffix = 1
    for bound in bboxes:
        print(f"  Sub-tile {suffix}")
        # Download a DEM array clipped to the sub-tile bbox, in the tiles' CRS.
        # Returns:
        #   mdt_arr  → raster array
        #   mdt_meta → raster metadata (profile)
        #   mdt_bbox → actual bbox used
        mdt_arr, mdt_meta, mdt_bbox = download_dem(bound, tiles.crs)

        # Compute and write DEM-derived rasters (e.g., slope/aspect).
        # The function itself is expected to handle naming and writing
        # into i_out_dir.
        compute_dem_products(mdt_arr, mdt_meta, f"_{suffix}", i_out_dir)
        print("    Derived products computed successfully.")
        suffix += 1

# -----------------------------------------------------------------------------
# 4) Build and export a GeoDataFrame with all sub-tiles, carrying parent attrs
# -----------------------------------------------------------------------------
# Create a dict suitable for building a GeoDataFrame from the per-tile
# sub-tile boxes.
divided_tiles_dict = create_divided_tile_gpd(
    divided_tiles_boxes, tiles["name"].to_list()
)

# Assemble the final GeoDataFrame with the same CRS as the input tiles.
divided_tiles_gpd = gpd.GeoDataFrame(divided_tiles_dict, crs=tiles.crs)

# Join the parent tile attributes onto each sub-tile by the "name" key.
# We use rsuffix to avoid clobbering the sub-tile geometry column.
divided_tiles_gpd = divided_tiles_gpd.join(
    tiles.set_index("name"), on="name", rsuffix="_duplicated"
)

# Drop the duplicate geometry from the joined attributes and write to disk.
# The output contains sub-tile polygons plus all original tile attributes.
divided_tiles_gpd.drop(columns=["geometry_duplicated"]).to_file(
    Path(ROOT, "data/divided_tiles_by_area.gpkg")
)

