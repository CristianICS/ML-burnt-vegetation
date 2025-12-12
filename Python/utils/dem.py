from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
import requests
from osgeo import gdal
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, box

BBox = Tuple[float, float, float, float]

def split_tile_by_area(
    tile_row,
    target_area_ha: float
) -> Tuple[List[BBox], List[Polygon]]:
    """
    Split a (potentially large) tile into square sub-tiles of ~target_area_ha.

    Returns:
        (bboxes_xyxy, bboxes_polygons)

    Notes:
        - `tile_row` is a GeoPandas row (e.g., from iterrows) with
          a `.geometry`.
        - If the tile is smaller than the target area, the original bbox
          is returned.
        - Output bbox coordinates are in the same CRS units as the input 
          geometry.
    """
    geom = tile_row.geometry
    minx, miny, maxx, maxy = geom.bounds

    target_area_m2 = target_area_ha * 10_000.0
    tile_size = math.sqrt(target_area_m2)

    # If the tile is already small, return its own bbox.
    if geom.area <= target_area_m2:
        bbox = (minx, miny, maxx, maxy)
        return [bbox], [box(*bbox)]

    # Regular grid of square bboxes across the tile extent.
    small_boxes: List[BBox] = []
    y = miny
    while y < maxy:
        x = minx
        ymax = min(y + tile_size, maxy)
        while x < maxx:
            xmax = min(x + tile_size, maxx)
            small_boxes.append((x, y, xmax, ymax))
            x += tile_size
        y += tile_size

    # Convert to polygons; clip to original polygon if you need strict
    # conformity.
    small_boxes_polygons = [
        box(x0, y0, x1, y1)
        for x0, y0, x1, y1 in small_boxes
    ]
    return small_boxes, small_boxes_polygons

def create_divided_tile_gpd(
    boxes_by_tile: Sequence[Iterable[Polygon]],
    tile_names: Sequence[str],
) -> dict:
    """
    Flatten sub-tiles into a dict suitable for GeoDataFrame construction.

    Returns:
        {
          "name": [parent tile name per subtile],
          "subtile_fid": [1..N within each tile],
          "geometry": [Polygon per subtile]
        }
    """
    out = {"name": [], "subtile_fid": [], "geometry": []}

    for polys, tname in zip(boxes_by_tile, tile_names):
        for i, poly in enumerate(polys, start=1):
            out["name"].append(tname)
            out["subtile_fid"].append(i)
            out["geometry"].append(poly)

    return out

def download_dem(
    bounds_xyxy: BBox,
    crs
) -> Tuple[np.ndarray, dict, rasterio.coords.BoundingBox]:
    """
    Download a 5 m DEM for a bbox via the Spanish PNOA WCS (EPSG:25830).

    Args:
        bounds_xyxy: (xmin, ymin, xmax, ymax) in the provided `crs`.
        crs: rasterio CRS (or anything accepted by rasterio.crs.CRS.from_*).

    Returns:
        (dem_array, dem_meta, dem_bounds)
    """
    xmin, ymin, xmax, ymax = bounds_xyxy

    # Reproject request bounds to EPSG:25830 (axis-safe)
    dst_crs = rasterio.crs.CRS.from_epsg(25830)
    xmin, ymin, xmax, ymax = transform_bounds(
        crs, dst_crs, xmin, ymin, xmax, ymax, densify_pts=0, always_xy=True
    )

    # Build WCS request with params to avoid formatting/encoding issues
    base_url = "https://servicios.idee.es/wcs-inspire/mdt"
    params = {
        "service": "WCS",
        "request": "GetCoverage",
        "version": "2.0.1",
        "coverageId": "Elevacion25830_25",
        # requests can't send two 'subset' keys unless we pass a list
        "subset": [f"long({xmin},{xmax})", f"lat({ymin},{ymax})"],
        "format": "image/tiff",
    }

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"WCS request failed: {exc}") from exc

    # Read GeoTIFF directly from memory
    with rasterio.MemoryFile(BytesIO(response.content)) as memfile:
        with memfile.open() as ds:  # type: ignore[assignment]
            dem = ds.read(1)
            dem_meta = ds.meta.copy()
            dem_bounds = ds.bounds

    return dem, dem_meta, dem_bounds

def compute_dem_products(
    mdt_arr: np.ndarray,
    mdt_meta: dict,
    suffix: str,
    out_dir: Path
) -> None:
    """
    Write DEM to disk and compute slope, aspect, and hillshade with GDAL.

    Files created (if missing):
        mdt{suffix}.tif, slope{suffix}.tif, aspect{suffix}.tif, hillshade{suffix}.tif
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gdal.DontUseExceptions()

    # Write source DEM
    dem_path = Path(out_dir, f"mdt{suffix}.tif")
    meta = mdt_meta.copy()
    meta.update({"count": 1})  # ensure single-band
    with rasterio.open(dem_path, "w", **meta) as dst:
        dst.write(mdt_arr, 1)

    # Helper: run GDAL DEMProcessing only if output is missing
    def _ensure_dem_product(dst_path: Path, proc: str, **opts) -> None:
        if dst_path.exists():
            return
        ds = gdal.DEMProcessing(str(dst_path), str(dem_path), proc, **opts)
        if ds is not None:
            ds.FlushCache()
            ds = None  # close dataset

    # Slope (degrees)
    _ensure_dem_product(Path(out_dir, f"slope{suffix}.tif"), "slope", computeEdges=True)

    # Aspect (0–360); zeroForFlat=False to avoid flipping interpretations
    _ensure_dem_product(
        Path(out_dir, f"aspect{suffix}.tif"),
        "aspect",
        computeEdges=False,
        zeroForFlat=False,
    )

    # Hillshade (simple illumination settings)
    _ensure_dem_product(
        Path(out_dir, f"hillshade{suffix}.tif"),
        "hillshade",
        azimuth=180,
        altitude=45,
    )
