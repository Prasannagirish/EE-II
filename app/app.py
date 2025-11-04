import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="AI-Driven Solar Site Selection", layout="wide")

st.title("☀️ AI-Driven Site Selection for Sustainable Solar Farms")

data_path = "data/scored_parcels.csv"
gdf_path = "data/parcels.geojson"

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    gdf = gpd.read_file(gdf_path)
    return gdf.join(df)

gdf = load_data()

st.sidebar.header("Filters")
score_min = st.sidebar.slider("Min Suitability Score", 0.0, 1.0, 0.5, 0.05)
filtered = gdf[gdf["suitability_score"] >= score_min]

st.map(filtered)

st.subheader("Top Candidate Sites")
st.dataframe(filtered.sort_values("suitability_score", ascending=False).head(10)[["suitability_score","pred_energy"]])
