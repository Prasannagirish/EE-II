import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import numpy as np
import os, sys, subprocess, json, yaml
from shapely.geometry import Point

# ---------- PATHS ----------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SRC_DIR   = os.path.join(ROOT_DIR, "src")

# Primary data artifacts
PARCELS_GEOJSON = os.path.join(DATA_DIR, "export.geojson")  # produced by features.py
PARCELS_FEATS   = os.path.join(DATA_DIR, "parcels_features.parquet")
SCORED_FILE     = os.path.join(DATA_DIR, "scored_parcels.csv")
MODEL_BIN       = os.path.join(MODEL_DIR, "lgb_site_model.bin")
MODEL_TXT       = os.path.join(MODEL_DIR, "lgb_site_model.txt")
MODEL_COMPARISON= os.path.join(DATA_DIR, "model_comparison.csv")
CONFIG_PATH     = os.path.join(ROOT_DIR, "config.yaml")

st.set_page_config(page_title="AI-Driven Solar Site Selection - Trichy", layout="wide")

# ---------- HELPERS ----------
def _exists(path: str) -> bool:
    return path is not None and os.path.exists(path)

def _python():
    # Use the same interpreter running Streamlit
    return sys.executable

def run_script(script_name: str) -> str:
    script_path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(script_path):
        st.error(f"Script not found: {script_path}")
        st.stop()
    result = subprocess.run([_python(), script_path], capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"❌ {script_name} failed:\n\n{result.stderr}")
        st.stop()
    return result.stdout

def ensure_pipeline_outputs():
    # Step 1: features
    if not _exists(PARCELS_FEATS) or not _exists(PARCELS_GEOJSON):
        st.info("🔧 Generating features...")
        run_script("features.py")
    # Step 2: train
    if not (_exists(MODEL_BIN) or _exists(MODEL_TXT)):
        st.info("⚙️ Training model...")
        run_script("model_train.py")
    # Step 3: inference
    if not _exists(SCORED_FILE):
        st.info("🧠 Running inference...")
        run_script("inference.py")

@st.cache_data(show_spinner=False)
def load_parcels() -> gpd.GeoDataFrame:
    # Prefer GeoJSON; fallback to features parquet and build a geometry if needed
    if _exists(PARCELS_GEOJSON):
        gdf = gpd.read_file(PARCELS_GEOJSON)
    elif _exists(PARCELS_FEATS):
        df = pd.read_parquet(PARCELS_FEATS)
        # If geometry missing, try to build from lat/lon if present (rare)
        if "geometry" in df.columns and gpd.array.is_geometry(df["geometry"]).any():
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        else:
            # Dummy geometry: if lat/lon not present, create centroid placeholders at 0,0 (will be filtered later)
            lat = df["lat"] if "lat" in df.columns else pd.Series([0]*len(df))
            lon = df["lon"] if "lon" in df.columns else pd.Series([0]*len(df))
            geom = gpd.points_from_xy(lon, lat, crs="EPSG:4326")
            gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    else:
        raise FileNotFoundError("No parcels dataset found. Expected data/export.geojson or data/parcels_features.parquet")

    # Drop duplicate columns if any sneaked in
    gdf = gdf.loc[:, ~gdf.columns.duplicated()].copy()

    # Ensure a projected CRS for centroid calculations, then back to WGS84
    try:
        if gdf.crs is None:
            # Heuristic: most rasters in India are EPSG:4326; set explicitly
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf_wgs84 = gdf.to_crs(4326)
    except Exception:
        # If reproject fails, keep as is
        gdf_wgs84 = gdf

    return gdf_wgs84

@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    if _exists(SCORED_FILE):
        df = pd.read_csv(SCORED_FILE)
        # keep only relevant, drop duplicates
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df
    return pd.DataFrame()

def merge_geo_and_scores(gdf: gpd.GeoDataFrame, scored: pd.DataFrame) -> gpd.GeoDataFrame:
    if not scored.empty and "parcel_id" in scored.columns and "parcel_id" in gdf.columns:
        gdf = gdf.merge(scored, on="parcel_id", how="left", suffixes=("", "_pred"))
    else:
        # If no scores yet, create safe default score
        if "suitability_score" not in gdf.columns:
            # Use whichever irradiance name exists to build a quick proxy
            for ghi_col in ["NSRDB_GHI_mean", "GHI_mean"]:
                if ghi_col in gdf.columns:
                    ghi = gdf[ghi_col].astype(float)
                    gdf["suitability_score"] = (ghi - ghi.min()) / max(1e-9, (ghi.max() - ghi.min()))
                    break
            else:
                gdf["suitability_score"] = 0.0
    return gdf

def ensure_lat_lon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure accurate centroid computation and add lat/lon columns.
    Fixes 'Geometry is in a geographic CRS' warnings.
    """
    try:
        # Ensure a CRS exists
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)

        # Estimate the best UTM CRS for accurate centroid (meters)
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)

        # Compute centroids in projected space
        cent_utm = gdf_utm.geometry.centroid

        # Convert centroids back to WGS84 (lat/lon)
        cent_wgs84 = gpd.GeoSeries(cent_utm, crs=utm_crs).to_crs(4326)
        gdf["lon"] = cent_wgs84.x
        gdf["lat"] = cent_wgs84.y
    except Exception as e:
        print(f"⚠️ Centroid computation fallback: {e}")
        # Fallback: compute directly in geographic CRS (less precise)
        gdf["lon"] = gdf.geometry.centroid.x
        gdf["lat"] = gdf.geometry.centroid.y

    return gdf


def best_available(cols, candidates, default=None):
    for c in candidates:
        if c in cols:
            return c
    return default

def read_config_model():
    if _exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return (cfg.get("model") or "").strip()
        except Exception:
            return ""
    return ""

# ---------- UI ----------
st.title("☀️ AI-Driven Solar Site Selection — Tiruchirappalli (Trichy)")
st.markdown(
    "This interactive dashboard uses **AI + geospatial data** to identify optimal locations for "
    "**utility-scale solar** in **Trichy**. It combines solar radiation, terrain, and land-use features, "
    "and automatically benchmarks multiple ML models to pick the best performer."
)

tabs = st.tabs([
    "1️⃣ Overview",
    "2️⃣ Map & Rankings",
    "3️⃣ Model Comparison",
    "4️⃣ Executive Summary"
])

# ---------- TAB 1: OVERVIEW ----------
with tabs[0]:
    st.subheader("🔍 Project Overview")
    st.markdown("""
- **Objective:** Identify the most suitable parcels for sustainable solar energy generation.  
- **Region:** Tiruchirappalli (Trichy), Tamil Nadu.  
- **Data Layers:**  
  - 🌤️ Solar irradiance (NSRDB / PVOUT)  
  - 🏔️ Terrain (DEM → slope, aspect)  
  - 🏭 Land use (industrial, farmland, etc.)  
- **AI:** Gradient boosting baseline with auto-optimization across multiple models.
    """)

    st.markdown("#### 🚀 Pipeline Steps")
    st.code(
        "1) Extract features  →  2) Train AI  →  3) Inference  →  4) Visualize  →  5) Compare Models",
        language="bash"
    )

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("🔁 Run Full AI Pipeline", use_container_width=True):
            ensure_pipeline_outputs()
            st.success("✅ Pipeline completed successfully!")

    with c2:
        if st.button("🧪 Run Model Optimization (Benchmark)", use_container_width=True):
            out = run_script("model_optimize.py")
            st.success("✅ Model optimization complete. See the Model Comparison tab.")
            st.text(out[-1500:])  # tail for quick log glance

    with c3:
        if st.button("🔧 Retrain Using Best Model", use_container_width=True):
            # Ensure model_comparison exists; model_train.py will already use optuna, but we keep config updated
            if _exists(MODEL_COMPARISON):
                # model_optimize.py already writes config.yaml with best model
                st.info("Training with best model from model_comparison.csv (see config.yaml)...")
            out = run_script("model_train.py")
            st.success("✅ Retraining complete.")
            st.text(out[-1500:])

    # Quick status panel
    st.markdown("#### 📦 Artifacts Status")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Parcels GeoJSON", "✅" if _exists(PARCELS_GEOJSON) else "❌")
    s2.metric("Features Parquet", "✅" if _exists(PARCELS_FEATS) else "❌")
    s3.metric("Scores CSV",      "✅" if _exists(SCORED_FILE) else "❌")
    s4.metric("Trained Model",   "✅" if (_exists(MODEL_BIN) or _exists(MODEL_TXT)) else "❌")

# ---------- TAB 2: MAP & RANKINGS ----------
# --- TAB 2: MAP & RANKINGS ---
with tabs[1]:
    st.subheader("🗺️ Solar Suitability Map")

    # Load geodata
    try:
        gdf = load_parcels()
    except Exception as e:
        st.error(f"Failed to load parcels: {e}")
        st.stop()

    scored = load_scored()
    gdf = merge_geo_and_scores(gdf, scored)
    gdf = ensure_lat_lon(gdf)

    # Choose best available feature names
    ghi_col   = best_available(gdf.columns, ["NSRDB_GHI_mean", "GHI_mean"])
    slope_col = best_available(gdf.columns, ["mean_slope", "slope_mean"])
    score_col = best_available(gdf.columns, ["suitability_score"])

    # Default score handling
    if score_col is None:
        st.warning("Suitability scores not found; using normalized irradiance as a proxy.")
        if ghi_col:
            vals = gdf[ghi_col].astype(float)
            gdf["__score__"] = (vals - vals.min()) / max(1e-9, (vals.max() - vals.min()))
            score_col = "__score__"
        else:
            gdf["__score__"] = 0.0
            score_col = "__score__"

    gdf[score_col] = gdf[score_col].astype(float).clip(0, 1)

    # --- Identify Top 10 Parcels ---
    top10 = gdf.sort_values(score_col, ascending=False).head(10).copy()
    top10_names = top10["name"].fillna("Unnamed Parcel").tolist()

    # --- Map Layers ---
    base_layer = pdk.Layer(
        "ScatterplotLayer",
        data=gdf,
        get_position=["lon", "lat"],
        get_color=f"[255 * (1 - {score_col}), 255 * {score_col}, 0, 120]",
        get_radius=80,
        pickable=True,
    )

    # Optional overlay for Top 10
    show_top10 = st.checkbox("✨ Highlight Top 10 Sites on Map", value=True)

    layers = [base_layer]
    if show_top10 and not top10.empty:
        highlight_layer = pdk.Layer(
            "ScatterplotLayer",
            data=top10,
            get_position=["lon", "lat"],
            get_color="[0, 255, 255, 255]",  # cyan highlight
            get_radius=250,
            pickable=True,
            tooltip=True,
        )
        label_layer = pdk.Layer(
            "TextLayer",
            data=top10,
            get_position=["lon", "lat"],
            get_text="name",
            get_color="[255,255,255]",
            get_size=12,
            get_alignment_baseline="'bottom'",
        )
        layers.extend([highlight_layer, label_layer])

    # --- View State ---
    view_state = pdk.ViewState(
        latitude=float(np.nanmean(gdf["lat"])) if len(gdf) else 10.7905,
        longitude=float(np.nanmean(gdf["lon"])) if len(gdf) else 78.7047,
        zoom=11,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "text": "{name}\nLanduse: {landuse}\nScore: {" + score_col + "}"
        }
    ))

    # --- Table: Top 10 Sites ---
    st.markdown("### 📋 Top 10 Most Suitable Parcels")
    rename_map = {}
    if ghi_col: rename_map[ghi_col] = "Solar Irradiance (W/m²)"
    if slope_col: rename_map[slope_col] = "Average Slope (°)"
    rename_map[score_col] = "AI Suitability Score"
    pretty = gdf.rename(columns=rename_map)

    cols_to_display = ["parcel_id"]
    for c in ["name", "landuse", "Solar Irradiance (W/m²)", "Average Slope (°)", "AI Suitability Score"]:
        if c in pretty.columns:
            cols_to_display.append(c)
    cols_to_display = list(dict.fromkeys(cols_to_display))

    top_sites = pretty.sort_values("AI Suitability Score", ascending=False).head(10)
    st.dataframe(top_sites[cols_to_display], use_container_width=True)

    # --- Download Button ---
    csv_data = top_sites[cols_to_display].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Top 10 CSV",
        data=csv_data,
        file_name="top10_sites.csv",
        mime="text/csv"
    )

    # --- Display Notes ---
    if show_top10:
        st.info(
            f"Highlighted {len(top10)} top-performing parcels directly on the map "
            f"({', '.join(top10_names[:5])}{'...' if len(top10_names) > 5 else ''})."
        )
# ---------- TAB 3: MODEL COMPARISON ----------
with tabs[2]:
    st.header("📊 Model Comparison & Optimization")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🧪 Run Benchmark Now (model_optimize.py)", use_container_width=True):
            out = run_script("model_optimize.py")
            st.success("✅ Benchmark complete. Results updated.")
            st.text(out[-1500:])

    with c2:
        if st.button("🔧 Retrain Using Best Model (model_train.py)", use_container_width=True):
            out = run_script("model_train.py")
            st.success("✅ Retraining complete.")
            st.text(out[-1500:])

    st.markdown("---")
    if _exists(MODEL_COMPARISON):
        df = pd.read_csv(MODEL_COMPARISON)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        st.subheader("Leaderboard")
        st.dataframe(
            df.style.highlight_min(color='lightgreen', subset=["RMSE", "MAE"])
                     .highlight_max(color='lightblue', subset=["R2"]),
            use_container_width=True
        )
        try:
            st.bar_chart(df.set_index("Model")[["RMSE", "MAE", "R2"]])
        except Exception:
            pass

        best_row = df.sort_values("RMSE").iloc[0]
        st.success(f"🏆 Best Model: **{best_row['Model']}**  |  RMSE: {best_row['RMSE']:.4f}  |  R²: {best_row['R2']:.4f}")

        current_cfg = read_config_model()
        if current_cfg:
            st.caption(f"Current config.yaml model → **{current_cfg}** (auto-updated by optimizer)")
    else:
        st.warning("⚠️ No model_comparison.csv found. Click **Run Benchmark Now** to generate it.")

# ---------- TAB 4: EXECUTIVE SUMMARY ----------
with tabs[3]:
    st.subheader("📈 Executive Summary — Solar Potential Insights")

    try:
        gdf = load_parcels()
        scored = load_scored()
        gdf = merge_geo_and_scores(gdf, scored)
    except Exception as e:
        st.warning(f"Summary limited: {e}")
        gdf = None

    if gdf is None or len(gdf) == 0:
        st.warning("No data available. Run the pipeline from the Overview tab.")
    else:
        ghi_col   = best_available(gdf.columns, ["NSRDB_GHI_mean", "GHI_mean"])
        slope_col = best_available(gdf.columns, ["mean_slope", "slope_mean"])
        score_col = best_available(gdf.columns, ["suitability_score", "__score__"])

        avg_ghi   = float(gdf[ghi_col].mean()) if ghi_col else float("nan")
        avg_slope = float(gdf[slope_col].mean()) if slope_col else float("nan")
        avg_score = float(gdf[score_col].mean()) if score_col else float("nan")

        st.markdown(f"""
**Overall Solar Potential**  
- Average solar irradiance: **{avg_ghi:.1f} W/m²**  
- Average terrain slope: **{avg_slope:.1f}°**  
- Average AI suitability: **{avg_score:.2f}**
""")

        if score_col:
            top5 = gdf.sort_values(score_col, ascending=False).head(5).copy()
            rename_map = {}
            if ghi_col:   rename_map[ghi_col] = "Solar Irradiance (W/m²)"
            if slope_col: rename_map[slope_col] = "Average Slope (°)"
            rename_map[score_col] = "AI Suitability Score"
            top5 = top5.rename(columns=rename_map)

            cols = ["parcel_id"]
            for c in ["name", "landuse", "Solar Irradiance (W/m²)", "Average Slope (°)", "AI Suitability Score"]:
                if c in top5.columns:
                    cols.append(c)
            cols = list(dict.fromkeys(cols))
            st.markdown("### 🌍 Top 5 High-Potential Sites")
            st.table(top5[cols])

        st.markdown("### 🧭 Regional Interpretation")
        st.info("""
- **Southern & eastern outskirts** of Tiruchirappalli often score highest thanks to good exposure and flatter terrain.  
- **Industrial and farmland parcels** dominate high-suitability ranks, simplifying permitting and interconnection.  
- **Dense urban areas** trend lower due to shading and space constraints.
        """)

        st.markdown("### ⚡ Strategic Recommendations")
        st.success("""
✅ Focus on flat, open farmland or industrial plots for utility-scale PV.  
✅ Avoid steep or forested tracts; grading & clearing cost can erode returns.  
✅ Next: field verification for the Top-10, grid-tie feasibility, and ownership due diligence.
        """)

        st.markdown("---")
        st.caption("Generated by the AI-based Solar Suitability Engine © 2025")
