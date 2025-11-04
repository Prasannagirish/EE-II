import geopandas as gpd
import pandas as pd
from .zonal_stats import zonal_stats_raster
from .terrain import compute_slope
import rasterio.mask
import numpy as np
import os

def build_features(parcels_path, ghi_raster, dem_raster):
    parcels = gpd.read_file(parcels_path)
    parcels = zonal_stats_raster(parcels, ghi_raster, stat="mean")

    slope, src = compute_slope(dem_raster)
    slope_vals = []
    for _, row in parcels.iterrows():
        geom = [row.geometry]
        try:
            out, _ = rasterio.mask.mask(src, geom, crop=True)
            arr = out[0]
            arr = arr[arr != src.nodata]
            slope_vals.append(np.mean(arr))
        except Exception:
            slope_vals.append(np.nan)
    parcels["mean_slope"] = slope_vals

    # Example derived feature
    parcels["suitability_proxy"] = parcels["NSRDB_GHI_mean"] / (1 + parcels["mean_slope"])
    return parcels

if __name__ == "__main__":
    feats = build_features("data/parcels.geojson", "data/NSRDB_GHI.tif", "data/dem.tif")
    feats.to_parquet("data/parcels_features.parquet")
