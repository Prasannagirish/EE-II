import rasterio
import numpy as np
from scipy import ndimage

def compute_slope(dem_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        xres, yres = src.res
        dx = ndimage.sobel(dem, axis=1) / (8.0 * xres)
        dy = ndimage.sobel(dem, axis=0) / (8.0 * yres)
        slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    return slope, src
