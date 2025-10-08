"""Download SIOSE from the extension of each image tile."""
from rasterio.warp import transform_bounds # type: ignore
from rasterio.crs import CRS # type: ignore
from owslib.wms import WebMapService # type: ignore
from pathlib import Path
from io import BytesIO
import rasterio # type: ignore
import geopandas as gpd # type: ignore
import xml.etree.ElementTree as ET
import requests

ROOT = Path(__file__).resolve().parent.parent
# Output path to save SIOSE images
out_path = Path(ROOT, "data/siose")

tiles_path = Path(ROOT, "data/divided_tiles_by_area.gpkg")
tiles = gpd.read_file(tiles_path)
# Tiles are subdivided in smaller ones, dissolve to perform SIOSE download.
# tiles = tiles.dissolve("name", as_index=False)

# Initialize the WMS
service = ''.join(["https://servicios.idee.es/",
    "wms-inspire/ocupacion-suelo-historico?",
    "Service=WMS"])

wms = WebMapService(service, version="1.3.0")
siose_layer_names = [n for n in list(wms.contents) if n.startswith("siose")]
siose_years = [int(n[-4:]) for n in siose_layer_names]

class Siose:

    def __init__(self, service_url, wms):
        """
        :service_url: The url to access the WMS
        :wms: owslib.wms.WebMapService object
        """
        self.wms = wms
        self.wms_url = service_url

    def define_bounds(self, bounds, bounds_crs, target_crs = None):
        """
        Use this function in case the target bounds' CRS is not within the
        WMS CRS list.

        :bounds: minx, miny, maxx, maxy
        :bounds_crs: CRS of the bbox coordinates.
        :target_crs: CRS to transform the bounds.
        """
        # Transform image bounds into valid ones
        # This is the CRS format of the SIOSE layer to retrieve
        if target_crs != None:
            xmin, ymin, xmax, ymax = bounds
            bounds = transform_bounds(bounds_crs, CRS.from_epsg(target_crs),
                left=xmin,
                bottom=ymin,
                right=xmax,
                top=ymax)

            self.crs = target_crs
        else:
            self.crs = bounds_crs
        
        self.bounds = bounds

    def get_delta(self, target_resolution):
        """
        Compute Delta: The closest pixel resolution to the desired one
        which gets the final image dimensions in pixels.

        The applied style for the siose layers is displayed at scales higher than 1:150,000. So, the first step involves compute the image size at 1:150,000 scale.

        :target_resolution: The pixel size of the extracted SIOSE image. It
        should be on the bbox crs units.
        """
        scale_denominator = 50000
        dpi = 96 * 2
        meters_per_inch = (target_resolution * dpi) / scale_denominator

        # Compute resolution in meters/pixel
        res = scale_denominator * meters_per_inch / dpi

        # Calculate image dimensions
        minx, miny, maxx, maxy = self.bounds
        # Compute bounding box width and height in degrees
        bbox_width = maxx - minx
        bbox_height = maxy - miny

        # Compute integer dimensions
        int_width = round(bbox_width / res)
        int_height = round(bbox_height / res)
        print(bbox_width / res)
        print(bbox_height / res)
        # Compute the adjusted resolution to ensure integer dimensions
        adjusted_resolution_x = bbox_width / int_width
        adjusted_resolution_y = bbox_height / int_height

        # Choose the closest resolution to maintain aspect ratio consistency
        adjusted_resolution = min(adjusted_resolution_x, adjusted_resolution_y)

        # Recalculate final integer dimensions
        final_width = round(bbox_width / adjusted_resolution)
        final_height = round(bbox_height / adjusted_resolution)

        return adjusted_resolution, final_width, final_height
    
    def coords_to_pixels(self, x, y, resolution):
        """
        Converts geographic coordinates (EPSG:4326) to row and column indices
        in the image matrix.

        Use this function to perform a GetFeatureInfo request.

        Parameters:
        x, y : float  - Longitude and Latitude of the point to transform.
        minx, miny, maxx, maxy : float - Bounding box coordinates.
        resolution : float - Adjusted resolution (degrees per pixel).

        Returns:
        tuple - (row, col) pixel coordinates.
        """
        minx, miny, maxx, maxy = self.bounds
        # Compute column index (X-axis)
        col = (x - minx) / resolution

        # Compute row index (Y-axis)
        # (Flipped because images start from the top)
        row = (maxy - y) / resolution

        # Convert to integer indices
        return int(round(row)), int(round(col))

    def download(self, siose_year, target_resolution, out_path):
        """Deprecated download function. Construct query from scratch."""
        minx, miny, maxx, maxy = self.bounds
        _, tile_width, tile_height = self.get_delta(target_resolution)

        wms_query = ''.join([f"{self.wms_url}",
            # Options to perform a GetFeatureInfo (slow)
            # "&request=GetFeatureInfo",
            # Extraction coordinates 
            # f"&x={col1}&y={row1}",
            # "&info_format=application/json"
            # f"&query_layers=siose{siose_year}",
            
            # Options to perform a GetMap query
            "&request=GetMap",
            # Avoid GeoTiff format, it yields java head exceptions
            "&format=image/geotiff",
            f"&layers=siose{siose_year}",

            # Options for both process
            f"&version={self.wms.identification.version}",
            "&styles=codiige",
            f"&srs=epsg:{self.crs}",
            # Bounding box for map extent.
            # `minx,miny,maxx,maxy` in units of the SRS.
            f"&bbox={minx},{miny},{maxx},{maxy}"
            # Width of map output, in pixels.
            f"&width={tile_width}",
            # Height of map output, in pixels.
            f"&height={tile_height}"
        ])

        # Send request to WMS
        response = requests.get(wms_query)
        content_type = response.headers.get("Content-Type", "").lower()
        valid_fmts = ["image/", "application/octet-stream"]

        # Check response status
        if response.status_code != 200:
            error_text = f"{response.status_code}, {response.text}"
            raise ValueError(f"Error in WMS request: {error_text}")

        elif "xml" in content_type:
            print(wms_query)
            # Check if WMS returned a ServiceExceptionReport
            tree = ET.fromstring(response.content)
            exception = './/{http://www.opengis.net/ogc}ServiceException'
            exceptions = [exc.text for exc in tree.findall(exception)]
            raise RuntimeError(f"WMS Exceptions: {",".join(exceptions)}")

        elif any(fmt in content_type for fmt in valid_fmts):
            # If no exception, load with rasterio
            image_data = BytesIO(response.content)

            with rasterio.open(image_data) as src:
                arr = src.read()
                meta = src.meta

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(arr)

for i, tile in tiles.iterrows():
    print(f"Process tile {tile["name"]}, subtile {tile["subtile_fid"]}")
    tile_bbox = tile.geometry.bounds
    crs = tiles.crs.to_epsg()
    out_image = Path(out_path, f"{tile["name"]}subid{tile["subtile_fid"]}")

    for siose_year in siose_years:
        i_siose = Siose(service, wms)
        i_siose.define_bounds(tile_bbox, crs)
        out_image_suffix = f"year{siose_year}.tif"
        i_siose.download(siose_year, 20, str(out_image) + out_image_suffix)
