import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import os 

def zonal_stats_raster(parcel_gdf, raster_path, stat="mean"):
    """Compute zonal stats for parcels over raster (GHI, NDVI, etc.)"""
    results = []
    with rasterio.open(raster_path) as src:
        for _, row in parcel_gdf.iterrows():
            geom = [row.geometry]
            try:
                out_img, _ = rasterio.mask.mask(src, geom, crop=True)
                arr = out_img[0]
                arr = arr[arr != src.nodata]
                if arr.size > 0:
                    val = float(np.nanmean(arr)) if stat == "mean" else float(np.nanmedian(arr))
                else:
                    val = np.nan
            except Exception:
                val = np.nan
            results.append(val)
    parcel_gdf[f"{os.path.basename(raster_path).split('.')[0]}_{stat}"] = results
    return parcel_gdf
