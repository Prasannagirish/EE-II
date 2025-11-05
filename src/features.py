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

# ---------------- utility functions ----------------

def zonal_stats(parcels, raster_path, prefix):
    """
    Compute zonal mean and std of raster for each polygon (robust + safe version).
    Handles nodata correctly, preserves valid zero values.
    """
    mean_vals, std_vals = [], []

    if not os.path.exists(raster_path):
        print(f"⚠️ Missing raster: {raster_path}")
        parcels[f"{prefix}_mean"] = np.nan
        parcels[f"{prefix}_std"] = np.nan
        return parcels

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if parcels.crs != raster_crs:
            parcels = parcels.to_crs(raster_crs)

        nodata_val = src.nodata
        print(f"🔹 Sampling raster: {os.path.basename(raster_path)} (nodata={nodata_val})")

        for geom in parcels.geometry:
            try:
                masked, _ = rasterio.mask.mask(src, [mapping(geom)], crop=True)
                arr = masked[0].astype(float)

                # Filter out only NaN and nodata, keep valid zeros
                if nodata_val is not None and not np.isnan(nodata_val):
                    arr = arr[arr != nodata_val]
                arr = arr[np.isfinite(arr)]

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

    # Optional debug for first raster (usually GHI)
    if prefix.upper() in ["GHI", "PVOUT"]:
        valid_count = np.sum(~np.isnan(mean_vals))
        print(f"✅ {prefix} stats computed → valid parcels: {valid_count}/{len(parcels)}, mean={np.nanmean(mean_vals):.2f}")

    return parcels

def compute_slope_aspect(dem_path):
    """
    Generate slope/aspect rasters from DEM.
    Automatically reprojects DEM to UTM for accurate slope computation.
    Compatible with all pyproj/rasterio versions.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from pyproj import CRS

    # Step 1: Determine suitable UTM zone from DEM bounds
    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        bounds = src.bounds
        lon_center = (bounds.left + bounds.right) / 2
        lat_center = (bounds.top + bounds.bottom) / 2

        # Determine correct UTM EPSG code (Northern Hemisphere for India)
        utm_zone = int((lon_center + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if lat_center >= 0 else 32700 + utm_zone
        utm_crs = CRS.from_epsg(utm_epsg)

        # Reproject DEM to UTM if needed
        if src_crs != utm_crs:
            print(f"🗺️ Reprojecting DEM from {src_crs} → {utm_crs}")
            transform, width, height = calculate_default_transform(
                src_crs, utm_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({
                "crs": utm_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            reprojected_path = dem_path.replace(".tif", "_utm.tif")
            with rasterio.open(reprojected_path, "w", **meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.bilinear,
                )
            dem_path = reprojected_path

    # Step 2: Compute slope and aspect in projected (UTM) coordinates
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

    # Step 3: Save slope and aspect rasters
    slope_path = dem_path.replace(".tif", "_slope.tif")
    aspect_path = dem_path.replace(".tif", "_aspect.tif")

    with rasterio.open(slope_path, "w", **meta) as dst:
        dst.write(slope.astype(np.float32), 1)
    with rasterio.open(aspect_path, "w", **meta) as dst:
        dst.write(aspect.astype(np.float32), 1)

    print(f"✅ Slope & aspect computed in {utm_crs.to_authority()[1]} (UTM Zone {utm_zone})")
    return slope_path, aspect_path


def one_hot_landuse(df):
    if "landuse" not in df.columns:
        return df
    dummies = pd.get_dummies(df["landuse"].fillna("unknown"), prefix="land")
    df = pd.concat([df, dummies], axis=1)
    return df


# ---------------- MAIN ----------------

if __name__ == "__main__":
    parcels_path = os.path.join(DATA_DIR, "export.geojson")
    if not os.path.exists(parcels_path):
        raise FileNotFoundError("Missing export.geojson — ensure parcels are exported first.")

    parcels = gpd.read_file(parcels_path)
    print(f"✅ Loaded {len(parcels)} parcels")
    print(f"🗺️ Initial CRS: {parcels.crs}")

    rasters = {
        "GHI": "GHI.tif",
        "DNI": "DNI.tif",
        "DIF": "DIF.tif",
        "PVOUT": "PVOUT.tif",
        "TEMP": "TEMP.tif",
        "DEM": "dem.tif"
    }

    # ---------- TERRAIN PROCESSING ----------
    dem_path = os.path.join(DATA_DIR, rasters["DEM"])
    if os.path.exists(dem_path):
        print("🧭 Computing slope and aspect from DEM...")
        slope_path, aspect_path = compute_slope_aspect(dem_path)
        with rasterio.open(slope_path) as src:
            parcels = parcels.to_crs(src.crs)
        parcels = zonal_stats(parcels, slope_path, "slope")
        parcels = zonal_stats(parcels, aspect_path, "aspect")
    else:
        print("⚠️ DEM not found, skipping slope/aspect computation.")

    # ---------- SOLAR & CLIMATE FEATURES ----------
    for key, fname in rasters.items():
        if key == "DEM":
            continue
        rpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(rpath):
            print(f"⚠️ Missing raster: {rpath}")
            continue
        print(f"🌤️ Sampling raster: {fname}")
        parcels = zonal_stats(parcels, rpath, key)

    # ---------- SUITABILITY PROXY ----------
# ---------- SUITABILITY PROXY ----------
    ghi = parcels.get("GHI_mean", np.nan)
    slope = parcels.get("slope_mean", np.nan)
    temp = parcels.get("TEMP_mean", np.nan)

# Replace NaNs with global means for safety
    slope_safe = np.where(np.isnan(slope), np.nanmean(slope), slope)
    temp_safe = np.where(np.isnan(temp), np.nanmean(temp), temp)
    ghi_safe = np.where(np.isnan(ghi), np.nanmean(ghi), ghi)

# Compute suitability proxy
    slope_clipped = np.clip(slope_safe, 0, None)  # ✅ replaces .clip(lower=0)
    proxy = (ghi_safe / (1 + slope_clipped)) * (1 - 0.001 * np.abs(temp_safe - np.nanmean(temp_safe)))

# Normalize between 0–1
    proxy_norm = (proxy - np.nanmin(proxy)) / (np.nanmax(proxy) - np.nanmin(proxy))
    parcels["suitability_proxy"] = proxy_norm

    print(f"✅ Suitability proxy computed: mean={np.nanmean(proxy_norm):.3f}, range=({np.nanmin(proxy_norm):.3f}, {np.nanmax(proxy_norm):.3f})")


    # ---------- LANDUSE ENCODING ----------
    parcels = one_hot_landuse(parcels)

    # ---------- SANITIZATION ----------
    for col in parcels.columns:
        if col == "geometry":
            continue
        try:
            if parcels[col].dtype == object or str(parcels[col].dtype) == "object":
                parcels[col] = parcels[col].astype(str)
        except Exception:
            parcels[col] = parcels[col].astype(str)

    # ---------- ASSIGN IDS ----------
    parcels["parcel_id"] = np.arange(1, len(parcels) + 1)
    parcels = parcels.loc[:, ~parcels.columns.duplicated()].copy()

    # ---------- SAVE OUTPUTS ----------
    out_parquet = os.path.join(DATA_DIR, "parcels_features.parquet")
    out_geojson = os.path.join(DATA_DIR, "export.geojson")

    parcels.to_parquet(out_parquet, index=False)
    parcels.to_file(out_geojson, driver="GeoJSON")

    print(f"✅ Features saved → {out_parquet}")
    print(f"✅ Updated GeoJSON → {out_geojson}")

    # ---------- SUMMARY ----------
    feature_cols = [c for c in parcels.columns if c.endswith("_mean")]
    print(f"🧠 Feature columns: {feature_cols}")
    print(parcels[feature_cols].describe())
    print("🎯 Completed feature extraction successfully.")
