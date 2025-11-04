import geopandas as gpd
import requests
import os

def load_parcels(path="data/parcels.geojson"):
    """Load parcel boundaries (GeoJSON or shapefile)."""
    return gpd.read_file(path)

def ensure_data_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

if __name__ == "__main__":
    ensure_data_dirs()
    parcels = load_parcels()
    print(f"Loaded {len(parcels)} parcels.")
