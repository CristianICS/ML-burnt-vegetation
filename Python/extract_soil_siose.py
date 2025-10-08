"""
Automatic extraction of sparse-vegetation (bare soil) points.

Pipeline (per tile and year)
1) Build mean summer composite and its NDVI.
2) Keep low-NDVI pixels (bare soil candidates) and thin them by distance.
3) Compute IL (illumination) and retain well-lit points (IL > 0.7).
4) Sample SIOSE RGB codes and keep only bare-soil classes.
5) Append results to a GeoPackage.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import rasterio  # type: ignore
from rasterio.io import MemoryFile  # type: ignore
from scipy.spatial import KDTree  # type: ignore
from shapely.geometry import Point  # type: ignore

from utils_tile import Tile, Composite
from utils_il import IL


# ------------------------------- Config --------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Output path (parent dir will be created if missing)
OUT_PATH = ROOT / "data" / "labels" / "ground_points_db.gpkg"

# Divided tiles (created by create_dem.py)
TILES_PATH = ROOT / "data" / "divided_tiles_by_area.gpkg"
TILES = gpd.read_file(TILES_PATH)

# Harmonized images root
IMAGES_PATH = Path(ROOT, "HarmoPAF_time_series")

# DEM-derived products used by IL
ASPECT_VRT = ROOT / "data" / "predictor_variables" / "dem" / "aspect.vrt"
SLOPE_VRT = ROOT / "data" / "predictor_variables" / "dem" / "slope.vrt"

# SIOSE “bare soil” class RGB codes
VALID_CODIIGE = {
    "roquedo": [217, 214, 199],
    "temporalmente_desarbolado_por_incendios": [60, 80, 60], # not in Aragon
    "suelo_desnudo": [210, 242, 194],
}
SIOSE_YEARS = [2005, 2009, 2011, 2014]


# ------------------------------ Utilities ------------------------------------

def filter_by_distance(
    df: gpd.GeoDataFrame,
    distance: float = 200.0
) -> gpd.GeoDataFrame:
    """
    Greedy thinning: keep one point per KDTree neighborhood within `distance`.
    """
    if df.empty:
        return df

    coords = np.column_stack(
        (df.geometry.x.to_numpy(), df.geometry.y.to_numpy())
    )
    tree = KDTree(coords)
    neighbors = tree.query_ball_point(coords, r=distance)

    selected = np.zeros(len(coords), dtype=bool)
    visited = np.zeros(len(coords), dtype=bool)

    for i in range(len(coords)):
        if visited[i]:
            continue
        selected[i] = True
        visited[neighbors[i]] = True

    return df.loc[selected].copy()


def extract_ndvi_points(ndvi: np.ndarray, img_meta: dict) -> gpd.GeoDataFrame:
    """
    Convert NDVI raster to point samples at pixel centroids within
    a fixed NDVI range.
    """
    mask = (ndvi >= 0.08) & (ndvi <= 0.15)  # bare-soil candidates
    if not mask.any():
        return gpd.GeoDataFrame(geometry=[], crs=img_meta["crs"])

    rows, cols = np.where(mask)
    xs, ys = rasterio.transform.xy(img_meta["transform"], rows, cols)
    return gpd.GeoDataFrame(
        {"NDVI": ndvi[rows, cols]},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=img_meta["crs"],
    )


def _ensure_out_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_flat_array(samples) -> np.ndarray:
    """
    Rasterio sample() yields 1×1 arrays; flatten to 1D.
    """
    # samples may be a generator of arrays shaped (bands,) or (1,) or (bands,1)
    return np.asarray(
        [np.squeeze(s).item()
         if np.asarray(s).size == 1
         else np.squeeze(s) for s in samples
        ]
    )


# --------------------------------- Main --------------------------------------

def main() -> None:
    _ensure_out_parent(OUT_PATH)

    for _, tile in TILES.iterrows():
        print(f'Process tile {tile["name"]}, subtile {tile["subtile_fid"]}')

        tile_obj = Tile(IMAGES_PATH / tile["name"])
        tile_years = set(tile_obj.get_years())

        for year in SIOSE_YEARS:
            if year not in tile_years:
                continue

            # 1) Composite & NDVI (summer window to reduce shadows)
            start, end = f"{year}-06-01", f"{year}-07-31"
            img_props = tile_obj.filter_date(start, end)
            bbox = tile.geometry.bounds
            images = Composite(img_props, tile_obj.imgs_dir, bbox, TILES.crs)
            images.compute_mean()
            images.compute_ndvi()

            # 2) NDVI → points, then distance thinning
            ndvi_points = extract_ndvi_points(
                images.composite[-1, :, :],
                images.composite_meta
            )
            if ndvi_points.empty:
                continue

            ndvi_points = filter_by_distance(ndvi_points, distance=200.0)
            if ndvi_points.empty:
                continue

            ndvi_points.loc[:, "YEAR"] = year

            # 3) IL computation and filtering
            il = IL(img_props, images.bounds, images.composite_meta["crs"])
            il_array = il.compute(ASPECT_VRT, SLOPE_VRT)

            il_meta = images.composite_meta.copy()
            il_meta.update({"count": 1, "dtype": il_array.dtype})

            with MemoryFile() as memfile:
                with memfile.open(**il_meta) as ds:
                    ds.write(il_array[None, :, :])  # add band axis

                    pts_reproj = ndvi_points.to_crs(ds.meta["crs"]).geometry
                    coords = np.column_stack(
                        (pts_reproj.x.to_numpy(), pts_reproj.y.to_numpy())
                    )
                    il_vals = _to_flat_array(ds.sample(coords, indexes=1))

            ndvi_points.loc[:, "IL"] = il_vals
            ndvi_points = ndvi_points.query("IL > 0.7")
            if ndvi_points.empty:
                continue

            # 4) SIOSE sampling
            siose_path = ROOT / "data" / "siose" / tile["name"] / f"siose{year}.tif"
            with rasterio.open(siose_path) as src:
                pts_reproj = ndvi_points.to_crs(src.meta["crs"]).geometry
                coords = np.column_stack((pts_reproj.x.to_numpy(), pts_reproj.y.to_numpy()))
                siose_samples = list(src.sample(coords))  # returns list of arrays length=3 (RGB)
            siose_arr = np.asarray(siose_samples)
            ndvi_points[["R", "G", "B"]] = siose_arr

            # 5) Filter by SIOSE bare-soil classes and write
            for siose_code, (r, g, b) in VALID_CODIIGE.items():
                mask = (ndvi_points["R"] == r) & (ndvi_points["G"] == g) & (ndvi_points["B"] == b)
                ground = ndvi_points.loc[mask].copy()
                if ground.empty:
                    continue

                ground.drop(columns=["R", "G", "B"], inplace=True)
                ground["siose_codiige"] = siose_code

                if OUT_PATH.exists():
                    ground.to_file(OUT_PATH, mode="a")
                else:
                    ground.to_file(OUT_PATH)


if __name__ == "__main__":
    main()
