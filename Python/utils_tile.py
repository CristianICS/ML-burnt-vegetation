from datetime import datetime
from pathlib import Path
from rasterio.windows import from_bounds # type: ignore
from rasterio.coords import BoundingBox # type: ignore
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject # type: ignore
from rasterio.warp import calculate_default_transform # type: ignore
from pyproj import Transformer
from shapely import Polygon # type: ignore
import rioxarray
import xarray as xr
import numpy as np
import rasterio # type: ignore
import geopandas as gpd # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
import warnings
import math

warnings.simplefilter("ignore", category=RuntimeWarning)

class Tile:

    # The images inside the following tiles have been processed
    # in order to obtain the same dimensions and bounding boxes.
    # Original ones are retained, and processed ones have '_crp` ext.`
    tiles_with_err = ['p5', 'p12', 'p4', 'p8_2', 'p6', 'p3', 'p7_2',
    'p21', 'p20', 'p25', 'p23', 'p7_4', 'p1_2', 'p24', 'p7_3', 'p19_3',
    'p16', 'p11_2', 'p19_2']

    def __init__(self, tile_path: Path):
        """
        Save the main properties of the tile.
        
        :tile_path: Directory containing the tile.
        """
        self.path = tile_path
        self.imgs_dir = Path(tile_path, "5_Harmonized")
        self.name = self.path.name

        # Guessing the image extension
        if self.name in self.tiles_with_err:
            self.imgs_extension = "_crp.tif"
        else:
            self.imgs_extension = ".tif"

        # Gathering img properties for the first time
        self.get_img_properties()
        # Check their attributes
        # self._check_img_meta()

        # Save image bands into the Tile
        self.band_names = [
            'coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        
        # Variable which will be switched to True if an error is found
        self.image_errors = False


    def get_img_properties(self):
        """Iterate through all tile images and store their properties."""
        # Init dictionary which will contain image properties.
        self.imgs_props = {}
        
        for img in sorted(self.imgs_dir.glob(f"*{self.imgs_extension}")):
            try:
                img_name = img.stem.split("_")
                # Retrieve properties from image name
                date_str = img_name[0] # Ex: "19840912_<...>.tif"
                date = datetime.strptime(date_str, "%Y%m%d")
                sensor_key = img_name[2]
            except:
                raise ValueError(f"Image name {img.stem} is not valid.")
            
            # Add properties to the image dictionary
            self.imgs_props[img.name] = {"date": date, "sensor": sensor_key}
    
    def get_years(self):
        """Extract all the years inside the tile."""
        return list(set([t['date'].year for t in self.imgs_props.values()]))

    def tif_to_zarr(self):
        """Too much computational time. Discard."""
        output_zarr = Path(self.path, "8_Zarr_files", self.name + ".zarr")
        datasets = []

        for fname, img_dict in self.imgs_props.items():

            fpath = Path(self.imgs_dir, fname)

            # Open raster as rioxarray
            da = rioxarray.open_rasterio(fpath, chunks={"x": 1024, "y": 1024})
            
            # Add temporal coordinate
            da = da.expand_dims(time=[img_dict["date"]])
            # da = da.assign_coords(time=[])

            datasets.append(da)

        if not datasets:
            print(f"No available TIFFs inside {self.imgs_dir}")
            return

        # Combine the data inside one array
        stack = xr.concat(datasets, dim="time")

        # Define encoding with compression (Blosc/Zstd is fast)
        encoding = {
            var: {
                "compressor": xr.backends.zarr.BloscCompressor(cname="zstd", clevel=3, shuffle=2),
                "chunks": (1, 1, 1024, 1024)  # (band, time, y, x)
            }
            for var in stack.data_vars
        }

        # Save to zarr compress array
        stack.to_zarr(
            output_zarr,
            mode="w",
            consolidated=True,
            encoding=encoding
        )

    def filter_date(self, start_date: str, end_date: str):
        """
        Select tile images by date.

        :start_date: Start date (included) "%Y-%m-%d"
        :end_date: End date (included) "%Y-%m-%d"
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        meta = self.imgs_props.items()
        filtered_dict = {
            # Note: This filter is possible because of datetime object inside
            # 'date' prop.
            k:v for (k,v) in meta if start <= v['date'] <= end
        }

        self.composite_props = filtered_dict

    def filter_years(self, years: list[int], months: list[int]):
        """
        Filter images based on a list of years and months.
        
        Keeps only images whose 'date' property matches any of the 
        combinations of the given years and months.
        """

        # Convert lists to sets for faster membership tests
        years_set = set(years)
        months_set = set(months)

        # Filter dictionary directly using date properties
        self.composite_props = {
            k: v for k, v in self.imgs_props.items()
            if v['date'].year in years_set
            and v['date'].month in months_set
        }
    
    def _check_img_meta(self):
        """
        Make sure the images have the same metadata and extension.
        
        UPGRADE: The borini images has been fixed. All the scenes have hte
        same metadata and bounds.
        """
        first_meta = None
        first_bounds =  None
        # Check if all the image have the same meta and bounds
        for img_name in self.imgs_props.keys():
            img_path = Path(self.imgs_dir, img_name)
            with rasterio.open(img_path) as src:
                meta = src.meta
                bounds = src.bounds
                if first_meta is None:
                    first_meta = meta
                    first_bounds = bounds
                # else:
                #     if meta != first_meta:
                #         self.image_errors = True
                #         warnings.warn("Invalid image metadata")
                #     if bounds != first_bounds:
                #         self.image_errors = True
                #         warnings.warn("Invalid bounds between image")

        self.img_meta = first_meta
        self.img_bounds = first_bounds
    
    def split_by_area(self, target_area_h) -> list:
        """
        Split the image tiles into smaller ones
        
        Avoid to split the tiles with an area less than "target_area_h".

        Return a GeoDataFrame of smaller bounding boxes that tile the original 
        geometry.

        :gdf: GeodataFrame.
        :target_area_h: Desired area of each small bounding box in hectars.
        """
        # Get bounds
        minx, miny, maxx, maxy = self.img_bounds

        target_area_m2 = target_area_h * 10000
        tile_size = math.sqrt(target_area_m2)
        # Create tiles
        small_boxes = []
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                xmax = min(x + tile_size, maxx)
                ymax = min(y + tile_size, maxy)
                small_boxes.append((x, y, xmax, ymax))

                x += tile_size

            y += tile_size

        return small_boxes

    def read(self, bbox: list = None, bbox_crs = None):
        """
        Store the images as numpy arrays.
        
        Arrays with shape (bands, rows, cols) / (bands, height, width).
        Compute the mean between all the rasters on a per-pixel basis,
        ignoring missing values (NAs).
        """
        if not hasattr(self, "composite_props"):
            raise ValueError("Filter the Tile object first.")

        if self.image_errors and (bbox is None):
            raise ValueError("A bbox is needed when images have errors.")
        
        if len(self.composite_props.keys()) == 0:
            return None, None, None

        # Store the numpy arrays inside a list
        img_arrays = []
        # Store the img array meta
        meta = None
        bounds = None

        if bbox is not None:
            # Transform the target bbox to the image CRS
            bbox = Aoi(bbox, bbox_crs)
            bbox.transform(self.img_meta["crs"])
            # Create a window based on bounds (minx, miny, maxx, maxy)
            window = from_bounds(*bbox.bbox, transform=src.transform)
        else:
            window = None
        
        for img_name in self.composite_props.keys():
            img_path = Path(self.imgs_dir, img_name)
            with rasterio.open(img_path) as src:

                if meta is None:
                    meta = self.img_meta

                if bounds is None:
                    bounds = src.bounds

                if window is not None:
                    # Store only the data inside the window
                    img_arrays.append(src.read(window=window))
                    meta["transform"] = src.window_transform(window)
                else:
                    img_arrays.append(src.read())

        if window is not None:
            meta.update({
                'height': window.height,
                'width': window.width
            })
            bounds = bbox.bbox

        # Stack the data inside a numpy array
        # Axis 0 will now represent the images, and axis 1 the band index
        array = np.stack(img_arrays, axis=0)
        array = np.where(array == meta['nodata'], np.nan, array)

        mean_arr, mean_meta = self.reduce_mean(array, meta)

        return mean_arr, mean_meta, bounds

    
    def reduce_mean(self, array, meta):
        """
        The median is good for not to take into account outliers.
        But there are some periods where the number of available images is
        low, so the median is not so well.

        Giving this reason, the mean stat is used to create image mosaics.
        """
        try:
            # Compute the median along the new axis
            arr = np.nanmean(array, axis=0)

        except:
            raise("Error computing the mean composite.")

        mean_data = np.where(arr == np.nan, meta["nodata"], arr)

        # Update metadata for output (now raster has 1 band and float values)
        mean_meta = meta.copy()
        mean_meta.update(count=mean_data.shape[0])
        
        return mean_data.astype(meta["dtype"]), mean_meta
    
    def read_xarr(self):
        
        if hasattr(self, "composite_props"):
            iterator = self.composite_props.items()
        # else:
            # iterator = self.imgs_props.items()

        rasters = []

        for name, d_dict in iterator:

            f = Path(self.imgs_dir, name)
            r = rioxarray.open_rasterio(f, chunks={"x": 1024, "y": 1024})
            # Add temporal dimension
            r = r.expand_dims(time=1)
            # Convert Python datetime to numpy.datetime64 to ensure
            # xarray can handle temporal slicing and operations correctly
            r = r.assign_coords(time=[np.datetime64(d_dict["date"])])
            rasters.append(r)

        stack = xr.concat(rasters, dim="time")
        return stack
    
    def xarr_subtract(self, pnts, mean):
        
        xs = np.array([p.x for p in pnts.geometry])
        ys = np.array([p.y for p in pnts.geometry])

        # Transform row/col by affine transform
        transform = mean.rio.transform()
        rows, cols = rowcol(transform, xs, ys)

        # Clip to valid indices
        # Avoid obtaining row/col values outside the arr shape
        rows = np.clip(rows, 0, mean.values.shape[1] - 1)
        cols = np.clip(cols, 0, mean.values.shape[2] - 1)

        # Extract values (bands, y, x) -> transpose to (points, bands)
        vals = mean.values[:, rows, cols].T # shape: n_points  n_band

        return vals


class Aoi:

    def __init__(self, bbox: list, bbox_crs):
        """
        Class to handle rasterio bbox.
        
        :bbox: [minx, miny, maxx, maxy]
        """
        self.bbox = bbox
        self.crs = bbox_crs

    def transform(self, dst_crs):
        lx, by, rx, ty = self.bbox
        # CRS, especially geographic ones (like EPSG:4326, WGS84), use
        # latitude/longitude (Y/X) order instead of the projected X/Y order.
        # Add the option "alwaysxy" to avoid errors.
        transformer = Transformer.from_crs(self.crs, dst_crs, always_xy=True)
        # Upper left
        lx, ty = transformer.transform(lx, ty)
        # Bottom right
        rx, by = transformer.transform(rx, by)
        self.bbox = [lx, by, rx, ty]
    
    def to_shapely(self):
        lx, by, rx, ty = self.bbox
        # Define bbox geometry as Shapely (mandatory to run mask function)
        geometry = [[lx,by], [lx,ty], [rx,ty], [rx,by]]
        # Mask functions needs a shapely object inside an iterable object
        return Polygon(geometry)

def reduce_by_season(
        tile: Tile,
        year: str,
        dates: list[str],
        season_name,
        bbox: list[float] = None,
        bbox_crs = None):
    """
    Create one image reducing all the image in a season.
     
    Season dates are created with year and dates.

    :year: Year to perform the composite.
    :dates: List with init and final composite month and day (mm-dd)
    :season_name: suffix to append the name of the season in all the bands.
    """
    # Filter the images inside the tile
    season_dates = [f'{int(year)}-{s}' for s in dates]
    tile.filter_date(*season_dates)

    if bbox is None:
        array, meta, bounds = tile.read(bbox, bbox_crs)
    else:
        array, meta, bounds = tile.read()



    # Interrupt the process due to there are no images
    if array is None:
        return (None,)*4

    # Add a prefix with the season to all img band names to distinguish them
    bnames = [f'{b}_{season_name}' for b in tile.band_names]
    return (array, bnames, meta, bounds)

def extract_data(gdf: gpd.GeoDataFrame, arr, meta, bnames, year, bounds):
    """
    Include in a dataframe with points the data within the array positions.

    The arr, meta, bnames and bounds parameters are obtained with Tile.read()

    The returned dataframe has the same rows and columns plus the extracted
    data.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined.")
    if "YEAR" not in gdf.columns:
        raise ValueError("GeoDataFrame must have a 'YEAR' column.")

    # Project points into raster CRS
    pnts = gdf.to_crs(meta["crs"])
    sindex = pnts.sindex

    # Create output column(s) if missing
    # Important: If one column is missing, the rest of them too
    if not all(name in pnts.columns.to_list() for name in bnames):
        pnts[bnames] = meta["nodata"]

    # Construct the mask to select rows to fill (True values)
    # Numpy filter faster than query approach
    year_mask = (pnts["YEAR"].to_numpy() == year)
    data_mask = (pnts[bnames].to_numpy() == meta["nodata"])
    # Expand year mask (n_rows,) to match data_mask (n_rows, n_cols)
    # Important: the order must be first data and then year mask
    mask = data_mask & year_mask[:, None]
    
    # Filter points for this year that don't yet have values
    if not mask.any(axis=None):
        return gdf

    # Instead of using rasterio MemoryFile, less efficient because 
    # the array must be written inside a temporary GeoTIFF, the rows
    # and cols are extracted by indexing. It's faster because the
    # serialization is avoided.
    # Although, the MemoryFile method is more secure and clearer.
    height, width = arr.shape[1], arr.shape[2]
    transform = meta["transform"]

    # Select only candidate points from current year subset
    # Collapse the mask to 1D by requiring all True in each row.
    # Operate along columns for each row, return True if all values in 
    # that row are True, otherwise False.
    reduced_mask = mask.all(axis=1)
    mask_idx = pnts.index[reduced_mask]
    minx, miny, maxx, maxy = bounds
    candidate_idx = list(
        set(mask_idx) & set(sindex.intersection((minx, miny, maxx, maxy)))
    )

    if not candidate_idx:
        return gdf

    subset = pnts.loc[candidate_idx]
    xs, ys = subset.geometry.x.values, subset.geometry.y.values

    # Convert coords to row/col indices
    cols, rows = ~transform * (xs, ys)
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)

    # Sample array directly
    for row, col, idx in zip(rows, cols, candidate_idx):
        # The points are inside the tile bbox
        if 0 <= row < height and 0 <= col < width:
            pixel_vals = arr[:, row, col]
            # Include the values in the original dataframe
            gdf.loc[idx, bnames] = tuple(float(v) for v in pixel_vals)

    return gdf


class Composite:

    def __init__(self, fire: gpd.GeoDataFrame, target_crs = 32630):
        """Extract information of one composite inside current fire."""
        self.fire = fire.to_crs(f"EPSG:{target_crs}")
        self.target_crs = target_crs
        # The original perimeters have a high spatial resolution,
        # generalize them in order to speed up the process.
        aoi_buffer = fire.simplify(100, preserve_topology=False)
        # Construct a buffer to include control zone (unburned)
        # Its bounding box will be the area to extract the information.
        self.aoi = aoi_buffer.geometry.buffer(300).total_bounds

        # The clipped images are stored inside this object.
        self.arr_obj = {}

    def common_grid_params(self, img_meta):
        """
        Compute the grid parameters from fire bounds.

        Use image resolution to compute the new width and height and transform
        parameters.

        :img_meta: Metadata from rasterio object.
        """
        # Get the desired resolution,
        # the one captured from the images metadata
        trns = img_meta["transform"]
        target_res = (trns.a, trns.e) # x and y res

        dst_bbox = self.aoi
        # Compute transform and output shape for the output raster
        width = abs(int((dst_bbox[2] -dst_bbox[0]) / target_res[0]))
        height = abs(int((dst_bbox[3] -dst_bbox[1]) / target_res[1]))

        src_crs = img_meta["crs"]
        # Compute the area to extract the info from the images
        window = from_bounds(*dst_bbox, transform = trns)
        # Generate the new transformation object
        dst_trns, width, height = calculate_default_transform(
            src_crs, self.target_crs, window.width, window.height, *dst_bbox
        )

        self.dst_trns = dst_trns
        self.width = width
        self.height = height

    def reproject_to_common_grid(self, arr, src_meta, dtype):
        """Adjust image to the calculated common grid params."""
        # Compute the new image dimensions based on fire bounds.
        dst_array = np.empty(
            (src_meta["count"], self.height, self.width), dtype=dtype)
            
        reproject(
            source=arr,
            destination=dst_array,
            src_transform=src_meta["transform"],
            src_crs=src_meta["crs"],
            dst_transform=self.dst_trns,
            dst_crs=self.target_crs,
            resampling=Resampling.nearest,
            src_nodata=src_meta["nodata"],
            dst_nodata=src_meta["nodata"]
        )
        
        # Replace NA number with na explicitly. The goal is avoid to
        # include these pixels to the model.
        dst_array = np.where(
            dst_array == src_meta['nodata'], np.nan, dst_array)

        # Return array and metadata for saving
        dst_meta = src_meta.copy()
        dst_meta.update({
            "crs": self.target_crs,
            "transform": self.dst_trns,
            "width": self.width,
            "height": self.height
        })
        
        return (dst_array, dst_meta)
    
    def add_array(self, arr, arr_name, arr_bands, meta, dtype=np.float64):
        """Apply the transformation to adjust the array to fire bounds."""
        dst_arr, dst_meta = self.reproject_to_common_grid(arr, meta, dtype)
  
        # Include the array in the current object
        self.arr_obj[arr_name] = {
            'arr': dst_arr, 'names': arr_bands, 'meta': dst_meta}

    def add_image(self, img_path, img_name, img_bands):
        """Open an image and apply the transformations to adjust fire bounds"""
        with rasterio.open(img_path) as src:
            arr = src.read(1)
            meta = src.meta
            dtype = src.dtypes[0]

        self.add_array(arr, img_name, img_bands, meta, dtype)

    def check_layers(self, names):
        """Check if certain layers have been inserted."""
        # Prove if all target names are inside the object dictionary
        if all(el in self.arr_obj.keys() for el in names):
            return True
        else:
            return False

    def concatenate(self):
        """Merge all the arrays contained in the object inside a new array."""
        # Merge band from all the array names
        bn = [item for l in self.arr_obj.values() for item in l['names']]
        # Store all the arrays inside a list
        arrs = [l['arr'] for l in self.arr_obj.values()]
        if len(arrs) == 0:
            raise ValueError("There is no classified image.")
        else:
            return (np.concatenate(arrs, axis=0), bn)

def ppconditional(cls, df):
    """Fix wrong Pinus pinaster locations."""
    # Store all the required info inside the axis 1 (None creates a new axis)
    values = np.hstack([
        cls[:, None],
        df["ELEVATION"].to_numpy()[:, None], df["ACIBASI"].to_numpy()[:, None]
    ]) # shape (rows*cols, 3)

    # Perform the conditional
    # 1. Switch "Pinus pinaster" to Pinus nigra
    ispn_mask = (values[:, 1] >= 850) & (values[:, 2] == 1)
    # Switch 13 to 11
    values[ispn_mask, 0] = np.where(
        values[ispn_mask, 0] == 13, 11, values[ispn_mask, 0])
    # 2. Switch "Pinus pinaster" to Pinus halepensis
    isph_mask = (values[:, 1] < 850) & (values[:, 2] == 1)
    # Switch 13 to 11
    # (in the reclass process, the nigra and halepensis values where merged)
    values[isph_mask, 0] = np.where(
        values[isph_mask, 0] == 13, 11, values[isph_mask, 0])

    return values[:, 0]

def d2sc_index(arr):
    """
    Compute D2SC
    
    It is the complement of the ratio between most voted and second voted
    class (Hermosilla et al., 2022, p. 5)

    - Find the maximum prob value (a) along the second axis (axis=1).
    - Find the second maximum value (b) along the second axis. A common way to 
      do this is by temporarily replacing the maximum value in each row with a
      very small value (or -inf), and then finding the new maximum.
    - Calculate the probability index: 100 * (1 - (b / a))

    :arr: Flattened array returned by predict_proba
    """
    # Step 1: Find the maximum value along the second axis
    a = np.nanmax(arr, axis=1)

    # Step 2: Find the second maximum value
    # Replace the max values with a very small number, then find the max again
    temp = np.where(arr == np.expand_dims(a, axis=1), -np.inf, arr)
    b = np.nanmax(temp, axis=1)

    # Step 3: Calculate the probability index
    probability_index = 100 * (1 - (b / a))

    return probability_index

def make_predictions(img_arr, img_bands, dst_meta, model_dict):
    # The rows,cols numpy arrays must be converted into 1D arrays
    # in order to insert the data into a pandas dataframe
    res_shape = img_arr.shape[1] * img_arr.shape[2]
    # Resize image into 1 dimension
    img_arr_1d = np.resize(img_arr, (len(img_bands), res_shape))

    # Save the position of the NA data. The predictions are made with
    # df without na data. Thanks to this mask, the valid data could be
    # inserted again in the 1D array.
    valid_mask = ~np.isnan(img_arr_1d).any(axis=0)
    # Matrix must be transposed in order to obtain cols*rows array
    img_df = pd.DataFrame(img_arr_1d.T, columns=img_bands)
    img_df.dropna(inplace=True)
    if img_df.shape[0] == 0:
        print("No data available to train")
        raise ValueError("No data available to train.")
    
    # Perform predictions
    with open(model_dict["src"], "rb") as f:
        model_pipe = joblib.load(f)

    cls = model_pipe.predict(img_df[model_dict["predictor_names"]])
    cls_cond = ppconditional(cls, img_df) # 1-D
    # Reconstruct the 1D output image, full of NA values
    cls_map = np.full((res_shape),
                        fill_value=dst_meta["nodata"],
                        # Use float64 in order to handle -9999 values
                        dtype=np.float64)
    # Add predicted data only in its original position
    cls_map[valid_mask] = cls_cond

    # Transform 1D array in the original image size
    cls_map_cond = np.resize(cls_map,
        (img_arr.shape[1], img_arr.shape[2]))
    
     # Obtain the probability values from the prediction
    cls_prob = model_pipe.predict_proba(img_df[model_dict["predictor_names"]])
    d2sc = d2sc_index(cls_prob)

    # The output of predict_proba is an arrary of 2 dimensions: ,npixels 
    # nlabels. Add the D2SC index.
    cls_prob = np.concatenate([cls_prob, d2sc[:, np.newaxis]], axis=1)
    # Reconstruct the 1D original image
    cls_map_prob_1d = np.full((res_shape, cls_prob.shape[1]),
                            fill_value = dst_meta["nodata"],
                            dtype=np.float64)
    cls_map_prob_1d[valid_mask] = cls_prob
    # Resize the 1D image to the original image size correctly
    # First, the (nrow*ncol, nclasses) matrix should be resized to
    # obtain a new one with (nclasses, nrow*ncols) shape (rasterio
    # format). Then reshape the image to match original one.
    cls_map_prob = cls_map_prob_1d.T.reshape(
        (cls_map_prob_1d.shape[1], img_arr.shape[1], img_arr.shape[2]))
    
    # Get the probability class names
    prob_bandnames = [f'prob_{cls}' for cls in model_pipe.classes_]
    prob_bandnames = prob_bandnames + ['D2SC']
    
    return cls_map_cond, cls_map_prob, prob_bandnames


def classify(composites_data, fire: gpd.GeoDataFrame, paths, model_dict):

    # Variables to store yearly classifications
    cls_dict = {}
    cls_prob_dict = {}
    cls_meta = False

    for year, season_dict in composites_data['years'].items():

        composite = Composite(fire)
        # Store the output image metadata
        img_meta = composites_data["meta"]
        # Create the common grid to crop all images
        composite.common_grid_params(img_meta)

        for season_name, season_data in season_dict.items():

            bnames = season_data['band_names']
            arr = season_data["array"]
            composite.add_array(arr, season_name, bnames, img_meta)

            # Check the number of pixels inside the fire
            n_pixels = arr.size
            na_total = np.count_nonzero(np.isnan(arr))
            availab = n_pixels - na_total
            # Compute the percentage of the available pixels (no na)
            pixel_perc = (availab * 100) / n_pixels

            if pixel_perc < 40:
                print(f"{year}: only {pixel_perc} percent of valid pixels.")
                raise ValueError("Less than 40 perc. of available pixels.")

        # Add supplementary information
        composite.add_image(paths["elev"], "elev", ["ELEVATION"])
        composite.add_image(paths["shadow"], "shad", ["SHADOW"])
        composite.add_image(paths["acibasi"], "acib", ["ACIBASI"])

        # Obtain the composite to make classifications
        try:
            img_arr, img_bands = composite.concatenate()
        except ValueError as e:
            print(e)
            continue

        # Classify image
        dst_meta = composite.arr_obj[season_name]["meta"]
        cls_arr, cls_prob_arr, prob_bands = make_predictions(
            img_arr, img_bands, dst_meta, model_dict)
        # Store the year classification in the above dict
        cls_dict[f'cls_{year}'] = cls_arr
        
        if type(cls_meta) == type(False):
            cls_meta = dst_meta

        # Store assign probability
        cls_prob_dict[year] = {'array': cls_prob_arr, 'band_names': prob_bands}

    return cls_dict, cls_prob_dict, cls_meta
