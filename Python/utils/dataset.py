"""
Handle dataset labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

# Project-specific helper
from .tile import Tile

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


def sample_image_into_points(
    gdf: gpd.GeoDataFrame,
    variable_name: str,
    img_path: Union[str, Path],
    band: int = 1,
) -> gpd.GeoDataFrame:
    """
    Sample a (single-band) raster at point locations and store values in
    a new column.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain point geometries with a valid CRS and include a 'YEAR' column.
    variable_name : str
        Name of the output column to create with sampled values.
    img_path : Path or str
        Path to the raster image.
    band : int, default 1
        Band index to sample.

    Returns
    -------
    GeoDataFrame
        A **copy** of the input GeoDataFrame with a new column containing
        sampled values.

    Notes
    -----
    - This function reprojects the points to the raster CRS prior to sampling.
    - If `variable_name` already exists, a ValueError is raised to avoid
    accidental overwrite.
    """
    img_path = Path(img_path)

    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined.")

    if variable_name in gdf.columns:
        raise ValueError(
            f"Column '{variable_name}' already exists; refusing to overwrite."
        )

    # Work on a copy to keep original untouched
    out = gdf.copy()

    with rasterio.open(img_path) as src:
        # Reproject points to raster CRS for correct sampling
        gdf_prj = out.to_crs(src.crs)

        # Ensure we are sampling a valid band
        if band not in src.indexes:
            raise ValueError(f"Band {band} not found in raster. Available: {list(src.indexes)}")

        # Extract values at point coordinates (x, y) in raster CRS
        coords = [(geom.x, geom.y) for geom in gdf_prj.geometry]
        # src.sample returns an iterable of arrays (shape (1,) for single band)
        raw_vals = list(src.sample(coords, indexes=band))

    # Flatten [[v], [v], ...] -> [v, v, ...]
    values = [float(v[0]) if len(v) else np.nan for v in raw_vals]
    out[variable_name] = values
    return out


class Landsat:
    """
    Handle Landsat-based seasonal predictor variables
    and the extraction for points that fall within tiles.
    """

    def __init__(
        self,
        dataset: gpd.GeoDataFrame,
        tile_bboxes: gpd.GeoDataFrame,
        out_folder: Union[str, Path],
        images_folder: Union[str, Path],
    ):
        """
        Parameters
        ----------
        dataset : GeoDataFrame
            Point dataset with a valid CRS and a 'YEAR' column.
        tile_bboxes : GeoDataFrame
            Polygon tile footprints with a 'name' column; must have a valid
            CRS.
        out_folder : Path or str
            Folder where per-tile outputs (.gpkg) will be written.
        images_folder : Path or str
            Folder containing per-tile imagery (folder structure expected
            by `Tile`).
        """
        if dataset.crs is None:
            raise ValueError("`dataset` must have a valid CRS.")
        if tile_bboxes.crs is None:
            raise ValueError("`tile_bboxes` must have a valid CRS.")

        self.seasons = {
            # Month-day windows (inclusive)
            # used to subset the xarray time dimension
            "spring": ["03-01", "05-31"],
            "summer": ["06-01", "08-31"],
            "summerlong": ["05-01", "08-31"],
        }

        self.images_folder = Path(images_folder)
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(parents=True, exist_ok=True)

        # Reproject tile footprints to match dataset CRS for spatial join
        if dataset.crs != tile_bboxes.crs:
            self.tile_bboxes = tile_bboxes.to_crs(dataset.crs)
        else:
            self.tile_bboxes = tile_bboxes

        # Keep only points that fall within any tile
        selected_pnts = dataset.sjoin(
            self.tile_bboxes, how="left", predicate="within")
        # Check for duplicated points (the same point within several tiles)
        if not selected_pnts.index.is_unique:
            # Move the current index (point ID) into a column:
            # avoiding duplicate indices using idxmax
            selected_pnts = (selected_pnts
                .reset_index().rename(columns={'index': 'pid'}))
            # Select the point inside the tile with more available years
            duration = selected_pnts["final_year"] - selected_pnts["init_year"]
            selected_pnts["duration"] = duration
            # Remove the ones without valid tile data
            selected_pnts.dropna(subset="duration", inplace=True)
            # For duplicated index groups, keep the row with
            # the maximum duration
            max_duration = selected_pnts.groupby("pid")["duration"].idxmax()
            selected_pnts = selected_pnts.loc[max_duration].set_index("pid")

        # Keep original dataset columns plus tile name (renamed to 'tile_name')
        base_cols = dataset.columns
        self.dataset = (
            selected_pnts
            .loc[:, list(base_cols) + ["name"]]
            .rename(columns={"name": "tile_name"})
        )

        # Discover already computed tiles: look for files named
        # 'tile_<name>.gpkg'
        self.computed_tiles = []
        for f in self.out_folder.glob("tile_*.gpkg"):
            stem = f.stem  # e.g., 'tile_ABC123'
            if stem.startswith("tile_") and len(stem) > 5:
                # everything after 'tile_'
                self.computed_tiles.append(stem[5:])
        print(f"Already computed tiles: {(', ').join(self.computed_tiles)}")

    def extract(self, tile_name: str) -> List[gpd.GeoDataFrame]:
        """
        Extract seasonal band statistics for all years present in the
        given tile.

        Returns
        -------
        List[GeoDataFrame]
            A list of per-year GeoDataFrames with the new seasonal band
            columns. Returns an empty list if there is nothing to process
            for the tile.
        """
        tdata: List[gpd.GeoDataFrame] = []

        tile_path = self.images_folder / tile_name
        tile = Tile(tile_path)

        # Points for this tile (must match exactly one tile name)
        pnts = self.dataset.query("tile_name == @tile_name").copy()
        if pnts.empty:
            return tdata

        pnts_years: np.ndarray = pd.unique(pnts["YEAR"])

        # Restrict tile composites to the years present in the points.
        # [3, 8] means restrict months March..August for
        # performance/filtering reasons.
        tile.filter_years(pnts_years, [3, 8])

        if len(tile.composite_props) == 0:
            print(f"[{tile_name}] No valid years found after filtering.")
            return tdata

        # Load multi-band xarray with a 'time' dimension and rioxarray accessor
        xarr = tile.read_xarr()

        # Normalize nodata to NaN for robust stats
        nodata_value = xarr.rio.nodata
        if nodata_value is not None:
            xarr = xarr.where(xarr != nodata_value, np.nan)

        # Reproject points to the image CRS prior to sampling
        if xarr.rio.crs != pnts.crs:
            pnts = pnts.to_crs(xarr.rio.crs)

        it_message = f"Extracting {tile_name} data by year"
        iterator = tqdm(pnts_years, desc=it_message) if TQDM else pnts_years

        for year in iterator:
            if year not in tile.get_years():
                continue

            year_df = pnts.query("YEAR == @year").reset_index(drop=True)

            # For each season, compute the temporal mean and sample the difference at points
            for sname, (start_md, end_md) in self.seasons.items():
                # Slice xarray time between YYYY-MM-DD bounds (inclusive)
                start_date = np.datetime64(f"{int(year)}-{start_md}")
                end_date = np.datetime64(f"{int(year)}-{end_md}")
                subset = xarr.sel(time=slice(start_date, end_date))

                # Mean over time -> still multi-band DataArray
                mean_da = subset.mean("time", skipna=True)

                # Sample delta (implementation provided by Tile)
                samples, mask = tile.xarr_subtract(mean_da, year_df)
                # shape: (n_points, n_bands)

                # Name the columns as <band>_<season>
                season_columns = [f"{b}_{sname}" for b in tile.band_names]
                bands_df = pd.DataFrame(
                    samples,
                    columns=season_columns,
                    index=year_df.loc[mask].index
                )
                year_df = year_df.join(bands_df)

            tdata.append(year_df)

        return tdata

    def batch_extraction(self) -> None:
        """
        Run extraction tile by tile and write one GeoPackage per tile.

        Each written file is named 'tile_<tile_name>.gpkg'.
        """
        for tname in self.tile_bboxes["name"].to_list():
            if tname in self.computed_tiles:
                # Skip tiles that have already been processed
                continue

            per_year_gdfs = self.extract(tname)
            if not per_year_gdfs:
                continue

            # Concatenate years into one GeoDataFrame (preserve geometry & CRS)
            # All items share the same CRS after extract()
            crs = per_year_gdfs[0].crs
            combined = gpd.GeoDataFrame(
                pd.concat(per_year_gdfs, ignore_index=True), crs=crs)

            # Write per-tile output
            outfile = self.out_folder / f"tile_{tname}.gpkg"
            combined.to_file(outfile, driver="GPKG", index=False)
            self.computed_tiles.append(tname)

    def merge_data(self) -> gpd.GeoDataFrame:
        """
        Merge all per-tile GeoPackages, remove duplicates (keeping rows with
        fewer NaNs), and unify to EPSG:4326.

        Returns
        -------
        GeoDataFrame
            Deduplicated dataset in EPSG:4326.
        """
        gpkgs = list(Path(self.out_folder).glob("*.gpkg"))
        if not gpkgs:
            raise FileNotFoundError(
                f"No .gpkg files found in '{self.out_folder}'.")
        
        frames = []
        crs = None
        # Reproject all the gpkgs into a common CRS
        for f in gpkgs:
            gdf = gpd.read_file(f)
            if gdf.crs is None:
                pass
            if crs is None:
                crs = gdf.crs.to_epsg()

            if gdf.crs.to_epsg() != crs:
                # Global CRS
                gdf = gdf.to_crs(crs)
            frames.append(gdf)

        # Concatenate while keeping geometry
        merged = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), crs=crs)

        original_columns = merged.columns

        # Remove geometry column to count the na data
        merged["_nulls"] = merged.drop(columns="geometry").isna().sum(axis=1)
        
        # Extract hashed-geometries (WKB bytes) for duplicate detection
        extract_geo = lambda g: None if g is None else g.wkb
        merged["_geomkey"] = merged.geometry.apply(extract_geo)
        
        # Define subset columns that must match to consider rows duplicates
        dp_cols = ["YEAR", "_geomkey"]
        dp_cols = dp_cols + (["source"] if "source" in merged.columns else [])

        # idx of best row per duplicate group (min nulls)
        # IMPORTANT: Do not preserve only rows with no na values.
        # The summer long period has data where the others not.
        best_idx = merged.groupby(dp_cols, sort=False)["_nulls"].idxmin()
        unique_dat = merged.loc[best_idx].copy()

        # Restore original column order
        unique_dat = unique_dat[original_columns]
        return unique_dat


def extract_global(
    dataset: gpd.GeoDataFrame,
    dem_variables_path: Union[str, Path],
    lit_variables_path: Union[str, Path],
) -> gpd.GeoDataFrame:
    """
    Extract global (non-temporal) predictor variables at point locations.

    Parameters
    ----------
    dataset : GeoDataFrame
        Point dataset with a valid CRS and a 'YEAR' column.
    dem_variables_path : Path or str
        Directory containing DEM-derived rasters (e.g., *.vrt).
    lit_variables_path : Path or str
        Directory containing lithological rasters (expects 'ACIBASI.tif').

    Returns
    -------
    GeoDataFrame
        Input dataset with additional columns for each sampled global variable.
    """
    dem_variables_path = Path(dem_variables_path)
    lit_variables_path = Path(lit_variables_path)

    out = dataset.copy()

    # DEM-derived variables
    # (each .vrt is a single-band mosaic or virtual raster)
    for p in sorted(dem_variables_path.glob("*.vrt")):
        variable = p.stem  # e.g., 'slope', 'aspect', ...
        print(f"Extracting DEM variable: {variable}")
        out = sample_image_into_points(out, variable, p)

    # Lithology
    acibasi = lit_variables_path / "ACIBASI.tif"
    if acibasi.exists():
        print("Extracting lithology variable: acibasi")
        out = sample_image_into_points(out, "acibasi", acibasi)
    else:
        print(f"Warning: '{acibasi.name}' not found in {lit_variables_path}. Skipping.")

    return out
