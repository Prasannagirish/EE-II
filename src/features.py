import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# -------------- utility functions --------------

def zonal_stats(parcels, raster_path, prefix):
    """Compute zonal mean and std of raster for each polygon (quiet version)"""
    mean_vals, std_vals = [], []
    if not os.path.exists(raster_path):
        print(f"⚠️ Missing raster: {raster_path}")
        parcels[f"{prefix}_mean"] = np.nan
        parcels[f"{prefix}_std"] = np.nan
        return parcels

    with rasterio.open(raster_path) as src:
        for geom in parcels.geometry:
            try:
                masked, _ = rasterio.mask.mask(src, [mapping(geom)], crop=True)
                arr = masked[0].astype(float)
                arr = arr[arr != src.nodata]
                if arr.size > 0:
                    mean_vals.append(np.nanmean(arr))
                    std_vals.append(np.nanstd(arr))
                else:
                    mean_vals.append(np.nan)
                    std_vals.append(np.nan)
            except Exception:
                mean_vals.append(np.nan)
                std_vals.append(np.nan)

    parcels[f"{prefix}_mean"] = mean_vals
    parcels[f"{prefix}_std"] = std_vals
    return parcels


def compute_slope_aspect(dem_path):
    """Generate slope/aspect rasters from DEM"""
    import rasterio
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        transform = src.transform
        xres, yres = transform.a, -transform.e
        dem = np.nan_to_num(dem)
        dx, dy = np.gradient(dem, xres, yres)
        slope = np.degrees(np.arctan(np.hypot(dx, dy)))
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.where(aspect < 0, 360 + aspect, aspect)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1)

    slope_path = dem_path.replace(".tif", "_slope.tif")
    aspect_path = dem_path.replace(".tif", "_aspect.tif")
    with rasterio.open(slope_path, "w", **meta) as dst:
        dst.write(slope.astype(np.float32), 1)
    with rasterio.open(aspect_path, "w", **meta) as dst:
        dst.write(aspect.astype(np.float32), 1)
    return slope_path, aspect_path


def one_hot_landuse(df):
    if "landuse" not in df.columns:
        return df
    dummies = pd.get_dummies(df["landuse"].fillna("unknown"), prefix="land")
    df = pd.concat([df, dummies], axis=1)
    return df


# -------------- MAIN --------------

if __name__ == "__main__":
    parcels_path = os.path.join(DATA_DIR, "export.geojson")
    if not os.path.exists(parcels_path):
        raise FileNotFoundError("Missing export.geojson — ensure parcels are exported first.")

    parcels = gpd.read_file(parcels_path)
    print(f"✅ Loaded {len(parcels)} parcels")

    rasters = {
        "GHI": "GHI.tif",
        "DNI": "DNI.tif",
        "DIF": "DIF.tif",
        "PVOUT": "PVOUT.tif",
        "TEMP": "TEMP.tif",
        "DEM": "dem.tif"
    }

    # Align CRS
    for rfile in rasters.values():
        path = os.path.join(DATA_DIR, rfile)
        if os.path.exists(path):
            with rasterio.open(path) as src:
                if parcels.crs != src.crs:
                    parcels = parcels.to_crs(src.crs)
            break

    # Terrain from DEM
    dem_path = os.path.join(DATA_DIR, rasters["DEM"])
    if os.path.exists(dem_path):
        slope_path, aspect_path = compute_slope_aspect(dem_path)
        parcels = zonal_stats(parcels, slope_path, "slope")
        parcels = zonal_stats(parcels, aspect_path, "aspect")

    # Solar & climate rasters
    for key, fname in rasters.items():
        if key == "DEM": continue
        rpath = os.path.join(DATA_DIR, fname)
        parcels = zonal_stats(parcels, rpath, key)

    # Suitability proxy (robust)
    ghi = parcels.get("GHI_mean", np.nan)
    slope = parcels.get("slope_mean", np.nan)
    temp = parcels.get("TEMP_mean", np.nan)
    parcels["suitability_proxy"] = (ghi / (1 + slope.clip(lower=0))) * (1 - 0.001 * (temp - np.nanmean(temp)).abs())
    s = parcels["suitability_proxy"]
    parcels["suitability_proxy"] = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s))

    # Land use encoding
    parcels = one_hot_landuse(parcels)

    # Convert all object columns (except geometry) to string for Parquet safety
    for col in parcels.columns:
        if col == "geometry":
            continue
        try:
            if parcels[col].dtype == object or str(parcels[col].dtype) == "object":
                parcels[col] = parcels[col].astype(str)
        except Exception:
        # Fallback for columns with mixed types
            parcels[col] = parcels[col].astype(str)

    # Fill NA + assign IDs
    parcels = parcels.fillna(0)
    parcels["parcel_id"] = np.arange(1, len(parcels) + 1)

    before_cols = len(parcels.columns)
    parcels = parcels.loc[:, ~parcels.columns.duplicated()]
    after_cols = len(parcels.columns)

    if after_cols < before_cols:
        print(f"⚙️ Removed {before_cols - after_cols} duplicate columns before saving.")
    out_parquet = os.path.join(DATA_DIR, "parcels_features.parquet")
    out_geojson = os.path.join(DATA_DIR, "export.geojson")

    parcels.to_parquet(out_parquet, index=False)
    parcels.to_file(out_geojson, driver="GeoJSON")

    print(f"✅ Features saved → {out_parquet}")
    print(f"✅ Updated GeoJSON → {out_geojson}")
    print(f"🧠 Feature columns: {list(parcels.columns)}")
