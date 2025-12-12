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
from pyproj import Transformer
from rasterio.io import MemoryFile  # type: ignore
from scipy.spatial import KDTree  # type: ignore
from shapely.geometry import Point  # type: ignore

from utils.tile import Tile
from utils.il import IL


# ------------------------------- Config --------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Output path (parent dir will be created if missing)
OUT_PATH = ROOT / "data" / "labels" / "ground_points_db.gpkg"

# Divided tiles (created with utils.dem "create_divided_tile_gpd")
TILES_PATH = ROOT / "data" / "divided_tiles_by_area.gpkg"
TILES = gpd.read_file(TILES_PATH)

# Harmonized images root
IMAGES_PATH = Path(r"F:\Borini\harmoPAF\HarmoPAF_time_series")

# DEM-derived products used by IL
ASPECT_VRT = ROOT / "data" / "predictor_variables" / "dem" / "aspect.vrt"
SLOPE_VRT = ROOT / "data" / "predictor_variables" / "dem" / "slope.vrt"

# SIOSE "bare soil" class RGB codes
VALID_CODIIGE = {
    "roquedo": [217, 214, 199],
    "temporalmente_desarbolado_por_incendios": [60, 80, 60], # not in Aragon
    "suelo_desnudo": [210, 242, 194],
}
SIOSE_YEARS = [2005, 2009, 2011, 2014]


# ------------------------------ Utilities ------------------------------------
def reproject_coords(pnts: gpd.GeoDataFrame, dst_crs):
    """Faster C approach to reproject points into dst_crs."""
    # Vectorized transformer (fast C code)
    transformer = Transformer.from_crs(
        pnts.crs,
        dst_crs,
        always_xy=True
    )

    xs = pnts.geometry.x.to_numpy()
    ys = pnts.geometry.y.to_numpy()
    # Reproject in vectorized form
    xs_r, ys_r = transformer.transform(xs, ys)
    return xs_r, ys_r

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
    Rasterio sample() yields 1x1 arrays; flatten to 1D.
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
        tile_years = tile_obj.get_years()

        for year in SIOSE_YEARS:
            if year not in tile_years:
                continue

            # Composite & NDVI (summer window to reduce shadows)
            start, end = f"{year}-06-01", f"{year}-07-31"
            tile_obj.filter_date(start, end)
            bbox = tile.geometry.bounds
            # Transform the GeoSeries tile into a GeoDataFrame
            tgdf = gpd.GeoDataFrame([tile], geometry="geometry", crs=TILES.crs)

            stack = tile_obj.read_xarr()
            reduced = tile_obj.reduce_xarr(
                stack,
                time_range=[start, end],
                reduce="median",   # "mean", "median", etc.
                gdf=tgdf
            )
            ndvi = tile_obj.compute_ndvi(reduced)
            # .expand_dims({"band": [0]})
            # NDVI -> points, then distance thinning
            ndvi_points = extract_ndvi_points(ndvi.values, tile_obj.img_meta)
            if ndvi_points.empty:
                continue

            ndvi_points = filter_by_distance(ndvi_points, distance=200.0)
            if ndvi_points.empty:
                continue

            ndvi_points.loc[:, "YEAR"] = year

            # IL computation and filtering
            il = IL(tile_obj.composite_props, bbox, TILES.crs)
            il_array = il.compute(ASPECT_VRT, SLOPE_VRT)

            il_meta = tile_obj.img_meta.copy()
            il_meta.update({"count": 1, "dtype": il_array.dtype})

            with MemoryFile() as memfile:
                with memfile.open(**il_meta) as ds:
                    ds.write(il_array[None, :, :])  # add band axis

                    xs_r, ys_r = reproject_coords(ndvi_points, il_meta["crs"])

                    # Build mask directly from coordinate bounds
                    left, bottom, right, top = ds.bounds
                    mask = (
                        (xs_r >= left) & (xs_r <= right) &
                        (ys_r >= bottom) & (ys_r <= top)
                    )

                    coords = np.column_stack((xs_r[mask], ys_r[mask]))
                    vals = ds.sample(coords, indexes=1)
                    il_vals = _to_flat_array(vals)

            ndvi_points.loc[mask, "IL"] = il_vals
            ndvi_points = ndvi_points.query("IL > 0.7")
            if ndvi_points.empty:
                continue

            # Extracting the Land Cover value from SIOSE WMS
            siose_path = ROOT / f"data/siose/{tile["name"]}/siose{year}.tif"
            with rasterio.open(siose_path) as src:
                xs_r, ys_r = reproject_coords(ndvi_points, src.meta["crs"])
                 # Build mask directly from coordinate bounds
                left, bottom, right, top = src.bounds
                mask = (
                    (xs_r >= left) & (xs_r <= right) &
                    (ys_r >= bottom) & (ys_r <= top)
                )
                coords = np.column_stack((xs_r[mask], ys_r[mask]))
                # Obtain a list of arrays with 3 bands (RGB)
                siose_samples = list(src.sample(coords))

            siose_arr = np.asarray(siose_samples)
            ndvi_points.loc[mask, ["R", "G", "B"]] = siose_arr

            # Filter by SIOSE bare-soil classes
            for siose_code, (r, g, b) in VALID_CODIIGE.items():
                mask = (
                    (ndvi_points["R"] == r) &
                    (ndvi_points["G"] == g) &
                    (ndvi_points["B"] == b)
                )
                ground = ndvi_points.loc[mask].copy()
                if ground.empty:
                    continue

                ground.drop(columns=["R", "G", "B"], inplace=True)
                ground["siose_codiige"] = siose_code

                if OUT_PATH.exists():
                    ground.to_file(OUT_PATH, mode="a", index=False)
                else:
                    ground.to_file(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
