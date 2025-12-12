r"""
Create an Incident Local angle (IL) image using scene solar geometry from GEE
and local terrain (aspect, slope) rasters.

$$
IL = \cos(\theta_z) \cdot \cos(\beta) 
   + \sin(\theta_z) \cdot \sin(\beta) \cdot \cos(\varphi_s - \alpha)
$$

where:
  $\theta_z$ = solar zenith,
  $\varphi_s$ = solar azimuth,
  $\beta$  = slope,
  $\alpha$  = aspect.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Iterable, Tuple

from Py6S import ExecutionError
import ee  # type: ignore
import numpy as np  # type: ignore
import rasterio  # type: ignore
import rioxarray
from rasterio.crs import CRS  # type: ignore
from rasterio.warp import transform_bounds  # type: ignore
from rasterio.windows import from_bounds


def _gee_init(project: str = "s-correction") -> None:
    """Initialize GEE once; ignore if already initialized."""
    try:
        ee.Initialize(project=project)
    except Exception:
        try:
            ee.Initialize()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to initialize Google Earth Engine.") from exc


class IL:
    """
    Compute Incident Local (IL) using GEE solar geometry + local DEM products.

    :bounds_xy: left (minx), bottom (miny), right (maxx), top (maxy)
    """

    # USGS Landsat L2 collections (C2/T1/L2)
    GEE_COLLECTIONS = {
        "L05": "LANDSAT/LT05/C02/T1_L2",
        "L07": "LANDSAT/LE07/C02/T1_L2",
        "L08": "LANDSAT/LC08/C02/T1_L2",
    }
    PROPS = ["SUN_AZIMUTH", "SUN_ELEVATION"]

    def __init__(
        self,
        tile_meta: Dict,
        bounds_xy: Tuple[float, float, float, float],
        crs
    ):
        _gee_init()
        self.prop_vals = self._extract_scene_metadata(
            tile_meta, bounds_xy, crs)
        self.bounds = bounds_xy
        self.crs = crs

    # ---------------------------- GEE helpers -----------------------------

    def _extract_scene_metadata(
        self,
        tile_meta: Dict,
        bounds_xy: Tuple[float, float, float, float],
        crs
    ) -> Dict[str, float]:
        """
        Average SUN_AZIMUTH and SUN_ELEVATION over all scenes intersecting
        the tile.

        Using the mean across scenes reduces border effects from varying 
        acquisition geometry between tiles.
        """
        xmin, ymin, xmax, ymax = transform_bounds(
            crs, CRS.from_epsg(4326), *bounds_xy
        )
        region = ee.Geometry.BBox(xmin, ymin, xmax, ymax)

        prop_vals = {p: [] for p in self.PROPS}

        for sensor, coll_id in self.GEE_COLLECTIONS.items():
            # Collect dates for this sensor from tile metadata
            dates = sorted(
                v["date"] for v in tile_meta.values() if v["sensor"] == sensor)
            if not dates:
                continue

            start = dates[0].strftime("%Y-%m-%d")
            # If only one date, extend the end by one month to ensure we hit 
            # the scene
            end = (
                ee.Date(dates[0].strftime("%Y-%m-%d")).advance(1, "month")
                if len(dates) == 1
                else dates[-1].strftime("%Y-%m-%d")
            )

            col = ee.ImageCollection(coll_id).filterDate(start, end).filterBounds(region)
            for p in self.PROPS:
                vals = col.aggregate_array(p).getInfo() or []
                prop_vals[p].extend(vals)

        # Fallback to NaN if a property list is empty to avoid 
        # ZeroDivisionError
        return {
            k: (mean(v) if v else float("nan"))
            for k, v in prop_vals.items()
        }

    # ----------------------------- IO helpers ----------------------------

    def _open_band(self, path) -> np.ndarray:
        """Read a single-band raster as a 2D NumPy array."""
        with rasterio.open(path) as src:

            # Open it only inside the desired bounds
            if self.crs != src.crs:
                # Transform the target bounds into image's crs
                xmin, ymin, xmax, ymax = transform_bounds(
                    self.crs, src.crs, *self.bounds, densify_pts=10
                )
            else:
                xmin, ymin, xmax, ymax = self.bounds
            # Clip bbox to raster bounds (avoid giant/outside windows)
            xmin = max(xmin, src.bounds.left)
            xmax = min(xmax, src.bounds.right)
            ymin = max(ymin, src.bounds.bottom)
            ymax = min(ymax, src.bounds.top)

            # If no overlap after clipping -> empty
            if xmin >= xmax or ymin >= ymax:
                raise ExecutionError(
                    "No overlap after clipping, returning empty array.")
                # return np.zeros((0, 0), dtype=src.dtypes[0])
            
            # Convert bounding box to raster window
            # and round them because rasterio's window expects integer values
            window = from_bounds(
                xmin, ymin, xmax, ymax, transform=src.transform
            ).round_offsets().round_lengths()
            # Read raster only inside prior window
            return src.read(1, window=window)

    # ------------------------------ Compute ------------------------------

    def compute(self, aspect_path, slope_path) -> np.ndarray:
        """
        Compute IL array (cosine of local incidence angle).
        Inputs:
            aspect_path : path to aspect raster (degrees, 0-360)
            slope_path  : path to slope  raster (degrees)
        Returns:
            2D NumPy array of IL values in [-1, 1] (typically >= 0 over lit slopes).
        """
        sun_az = float(self.prop_vals.get("SUN_AZIMUTH", float("nan")))
        sun_el = float(self.prop_vals.get("SUN_ELEVATION", float("nan")))

        # Load DEM products
        aspect_deg = self._open_band(aspect_path)
        slope_deg = self._open_band(slope_path)

        # Degrees to radians (use np.deg2rad for clarity/vectorization)
        aspect = np.deg2rad(aspect_deg)
        slope = np.deg2rad(slope_deg)
        theta_z = np.deg2rad(90.0 - sun_el)  # solar zenith
        phi_s = np.deg2rad(sun_az)           # solar azimuth

        # IL formula (no extra cos() — IL is already the cosine of local 
        # incidence angle)
        il = math.cos(theta_z) * np.cos(slope) + math.sin(theta_z) * np.sin(slope) * np.cos(phi_s - aspect)

        # Optional: clip tiny numeric drift
        return np.clip(il, -1.0, 1.0)
