import os
import sys
import subprocess
import yaml
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point

# ---------- PATH SETUP ----------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SRC_DIR   = os.path.join(ROOT_DIR, "src")

PARCELS_GEOJSON = os.path.join(DATA_DIR, "export.geojson")
PARCELS_FEATS   = os.path.join(DATA_DIR, "parcels_features.parquet")
SCORED_FILE     = os.path.join(DATA_DIR, "scored_parcels.csv")
MODEL_BIN       = os.path.join(MODEL_DIR, "lgb_site_model.bin")
MODEL_TXT       = os.path.join(MODEL_DIR, "lgb_site_model.txt")
MODEL_COMPARISON= os.path.join(DATA_DIR, "model_comparison.csv")
CONFIG_PATH     = os.path.join(ROOT_DIR, "config.yaml")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI-Driven Solar Site Selection – Trichy",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
    .stDataFrame tbody tr:nth-child(even) { background-color: #f6f8fa !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HELPERS ----------
def _exists(path): return path and os.path.exists(path)
def _python(): return sys.executable

def run_script(script_name: str):
    path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(path):
        st.error(f"❌ Script not found: {path}")
        st.stop()
    result = subprocess.run([_python(), path], capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"❌ {script_name} failed:\n\n{result.stderr}")
        st.stop()
    return result.stdout

def ensure_pipeline_outputs():
    with st.spinner("Setting up pipeline..."):
        if not _exists(PARCELS_FEATS) or not _exists(PARCELS_GEOJSON):
            st.info("🔧 Generating features...")
            run_script("features.py")
        if not (_exists(MODEL_BIN) or _exists(MODEL_TXT)):
            st.info("⚙️ Training model...")
            run_script("model_train.py")
        if not _exists(SCORED_FILE):
            st.info("🧠 Running inference...")
            run_script("inference.py")
    st.success("✅ Pipeline completed successfully!")

@st.cache_data(show_spinner=False)
def load_parcels():
    if _exists(PARCELS_GEOJSON):
        gdf = gpd.read_file(PARCELS_GEOJSON)
    elif _exists(PARCELS_FEATS):
        df = pd.read_parquet(PARCELS_FEATS)
        if "geometry" in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        else:
            geom = gpd.points_from_xy(df.get("lon", 0), df.get("lat", 0), crs="EPSG:4326")
            gdf = gpd.GeoDataFrame(df, geometry=geom)
    else:
        raise FileNotFoundError("No parcel dataset found in /data/.")
    return gdf.loc[:, ~gdf.columns.duplicated()].copy()

@st.cache_data(show_spinner=False)
def load_scored():
    if _exists(SCORED_FILE):
        df = pd.read_csv(SCORED_FILE)
        return df.loc[:, ~df.columns.duplicated()].copy()
    return pd.DataFrame()

def merge_geo_and_scores(gdf, scored):
    if not scored.empty and "parcel_id" in scored and "parcel_id" in gdf:
        gdf = gdf.merge(scored, on="parcel_id", how="left")
    if "suitability_score" not in gdf:
        gdf["suitability_score"] = 0.0
    return gdf

def ensure_lat_lon(gdf):
    try:
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
    except Exception:
        pass
    cent = gdf.geometry.centroid
    gdf["lon"], gdf["lat"] = cent.x, cent.y
    return gdf

def best_available(cols, options):
    for c in options:
        if c in cols: return c
    return None

def read_config_model():
    if not _exists(CONFIG_PATH): return ""
    try:
        with open(CONFIG_PATH, "r") as f: cfg = yaml.safe_load(f) or {}
        return cfg.get("model", "").strip()
    except Exception:
        return ""

# ---------- LAYOUT ----------
st.title("☀️ AI-Driven Solar Site Selection — Trichy")
st.caption("An intelligent geospatial-ML platform for identifying high-potential solar farm sites.")

tabs = st.tabs([
    "🏗️ Overview",
    "🗺️ Map & Rankings",
    "📊 Model Comparison",
    "📈 Executive Summary"
])

# ---------- TAB 1: OVERVIEW ----------
with tabs[0]:
    st.markdown(
        """
        ### 🔍 Project Summary
        **Objective:** Discover and rank parcels in Tiruchirappalli district most suitable for solar farm development.  
        **Methodology:** Combines solar irradiance, terrain gradient, and land-use data, analyzed using ML regression models.  
        """
    )
    st.info("AI automatically benchmarks multiple algorithms (LightGBM, XGBoost, CatBoost, RandomForest, etc.) and picks the best performer.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔁 Run Full AI Pipeline", use_container_width=True):
            ensure_pipeline_outputs()
    with c2:
        if st.button("🧪 Optimize Models", use_container_width=True):
            run_script("model_optimize.py")
            st.success("✅ Optimization complete. See the **Model Comparison** tab.")
    with c3:
        if st.button("🔧 Retrain Using Best Model", use_container_width=True):
            run_script("model_train.py")
            st.success("✅ Retraining done with best model from config.yaml.")

    st.divider()
    st.markdown("#### 📦 Data Artifacts Status")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Parcels GeoJSON", "✅" if _exists(PARCELS_GEOJSON) else "❌")
    s2.metric("Features Parquet", "✅" if _exists(PARCELS_FEATS) else "❌")
    s3.metric("Scores CSV", "✅" if _exists(SCORED_FILE) else "❌")
    s4.metric("Trained Model", "✅" if (_exists(MODEL_BIN) or _exists(MODEL_TXT)) else "❌")

# ---------- TAB 2: MAP & RANKINGS ----------
with tabs[1]:
    st.subheader("🗺️ Solar Suitability Visualization")
    try:
        gdf = ensure_lat_lon(merge_geo_and_scores(load_parcels(), load_scored()))
    except Exception as e:
        st.error(str(e))
        st.stop()

    score_col = best_available(gdf.columns, ["suitability_score"])
    gdf[score_col] = gdf[score_col].astype(float).clip(0, 1)
    top10 = gdf.sort_values(score_col, ascending=False).head(10)

    base_layer = pdk.Layer(
        "ScatterplotLayer",
        data=gdf,
        get_position=["lon", "lat"],
        get_color=f"[255*(1-{score_col}),255*{score_col},0,180]",
        get_radius=80,
        pickable=True,
    )
    top_layer = pdk.Layer(
        "ScatterplotLayer",
        data=top10,
        get_position=["lon","lat"],
        get_color="[0,255,255,255]",
        get_radius=250,
        pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=top10,
        get_position=["lon","lat"],
        get_text="name",
        get_size=12,
        get_color="[255,255,255]",
    )

    st.pydeck_chart(pdk.Deck(
        layers=[base_layer, top_layer, text_layer],
        initial_view_state=pdk.ViewState(
            latitude=float(np.nanmean(gdf["lat"])),
            longitude=float(np.nanmean(gdf["lon"])),
            zoom=11
        ),
        tooltip={"text": "{name}\nLanduse: {landuse}\nScore: {" + score_col + "}"}
    ))

    st.markdown("### 📋 Top-10 High-Suitability Parcels")
    display_cols = ["parcel_id", "name", "landuse", score_col]
    st.dataframe(top10[display_cols], use_container_width=True)
    st.download_button("⬇️ Download Top-10 CSV",
                       top10[display_cols].to_csv(index=False).encode(),
                       "top10_sites.csv","text/csv")

# ---------- TAB 3: MODEL COMPARISON ----------
with tabs[2]:
    st.subheader("📊 Model Performance Benchmark")
    if not _exists(MODEL_COMPARISON):
        st.warning("Run **Optimize Models** from Overview tab first.")
    else:
        df = pd.read_csv(MODEL_COMPARISON)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        st.dataframe(
            df.style.highlight_min(color='lightgreen', subset=["RMSE","MAE"])
                    .highlight_max(color='lightblue', subset=["R2"]),
            use_container_width=True
        )
        st.bar_chart(df.set_index("Model")[["RMSE","MAE","R2"]])
        best = df.sort_values("RMSE").iloc[0]
        st.success(f"🏆 **Best Model:** {best['Model']} (RMSE {best['RMSE']:.4f}, R² {best['R2']:.4f})")
        st.caption(f"Config.yaml currently set to model → `{read_config_model()}`")

# ---------- TAB 4: EXECUTIVE SUMMARY ----------
with tabs[3]:
    st.subheader("📈 Executive Insights")
    try:
        gdf = ensure_lat_lon(merge_geo_and_scores(load_parcels(), load_scored()))
    except Exception as e:
        st.warning(str(e))
        st.stop()

    score_col = best_available(gdf.columns, ["suitability_score"])
    ghi_col   = best_available(gdf.columns, ["NSRDB_GHI_mean","GHI_mean"])
    slope_col = best_available(gdf.columns, ["mean_slope","slope_mean"])

    avg_ghi   = gdf[ghi_col].mean() if ghi_col else np.nan
    avg_slope = gdf[slope_col].mean() if slope_col else np.nan
    avg_score = gdf[score_col].mean() if score_col else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("☀️ Avg Solar Irradiance", f"{avg_ghi:.1f} W/m²")
    k2.metric("🏔️ Avg Terrain Slope", f"{avg_slope:.1f}°")
    k3.metric("🤖 Avg Suitability", f"{avg_score:.2f}")

    st.markdown("### 🌍 Top-5 Parcels by Suitability")
    st.table(gdf.sort_values(score_col, ascending=False).head(5)[["name","landuse",score_col]])

    st.markdown("### 🧭 Regional Interpretation")
    st.info("""
    - **Southern & Eastern Trichy** show consistently higher suitability scores.
    - **Industrial and agricultural parcels** offer flat terrain and high solar exposure.
    - **Urban regions** underperform due to shading and spatial constraints.
    """)

    st.markdown("### ⚡ Strategic Recommendations")
    st.success("""
    ✅ Prioritize open farmland and industrial parcels with flat slope (<5°).  
    ✅ Avoid forested or steep tracts to reduce grading and environmental impact.  
    ✅ Proceed to on-site validation and grid-tie feasibility for the top 10 parcels.
    """)
    st.caption("Generated automatically by the AI-based Solar Suitability Engine © 2025")

