# ☀️ AI-Driven Solar Site Selection — Tiruchirappalli (Trichy)

An intelligent geospatial dashboard that identifies **optimal solar farm locations** using AI and GIS data.  
The project integrates **LightGBM, XGBoost, CatBoost, and RandomForest** models with **terrain, land-use, and solar irradiance layers** to predict a **solar suitability score** for each land parcel.

---

## 🚀 Key Features

### 🔹 End-to-End Pipeline
1. **Feature Extraction** — Raster statistics (DEM, irradiance, slope, aspect) aggregated by parcel polygons  
2. **Model Training** — Auto-optimized ML regressors benchmarked using Optuna  
3. **Inference** — Predicts suitability scores for each parcel  
4. **Visualization** — Interactive Streamlit dashboard with ranking, maps, and executive summary  
5. **Model Comparison** — Auto-selects best model and updates `config.yaml`

---

## 🗺️ Dashboard Highlights

### Tabs Overview:
| Tab | Description |
|------|--------------|
| **1️⃣ Overview** | Run feature extraction, training, and inference directly |
| **2️⃣ Map & Rankings** | Interactive map of parcels with color-coded suitability scores; highlight top 10 |
| **3️⃣ Model Comparison** | Displays RMSE/MAE/R² metrics for all models |
| **4️⃣ Executive Summary** | AI insights, top-5 summary, and regional interpretation |

### 📍 Visualization Example:
- Green = High Suitability  
- Yellow = Medium Suitability  
- Red = Low Suitability  
- Cyan Highlight = Top 10 Parcels

---

## 🧠 Machine Learning Stack

| Model | Framework | Description |
|--------|------------|--------------|
| LightGBM | `lightgbm` | Gradient boosting baseline |
| XGBoost | `xgboost` | Tree boosting with GPU support |
| CatBoost | `catboost` | Efficient categorical boosting |
| RandomForest | `scikit-learn` | Ensemble baseline |
| Ridge | `scikit-learn` | Linear regression baseline |
| MLPRegressor | `scikit-learn` | Neural network baseline |

Each model is evaluated on **RMSE**, **MAE**, and **R²** — best model auto-updates `config.yaml`.

---


## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Prasannagirish/EE-II.git
cd EE-II
```
### 2️⃣ Set up Python environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Usage
- Run the full pipeline manually
```bash
python src/features.py
python src/model_train.py
python src/inference.py
```
- or interactively via the Streamlit Dashboard 
```bash
streamlit run app/app.py
```
- Then open the URL shown (typically http://localhost:8501).


### 🧩 Model Optimization
- To benchmark and auto-select the best model:
``` bash 
python src/model_optimize.py
```
- This generates:
    - data/model_comparison.csv
    - Updates config.yaml with the best model (e.g. “randomforest”)

### 📈 Outputs
| File                        | Description                              |
| --------------------------- | ---------------------------------------- |
| `data/scored_parcels.csv`   | AI-predicted suitability for each parcel |
| `data/model_comparison.csv` | Model benchmark results                  |
| `models/lgb_site_model.bin` | Trained model artifact                   |


### 🧭 Interpreting Results
- Suitability Score: 0–1 (higher = better)
- Green/Yellow/Red Map: Indicates feasibility gradient
- Top 10 Parcels: Highlighted on map and downloadable as CSV
- Executive Summary: Simplified view for non-technical users

### 🧭 Acknowledgements
- NASA NSRDB for solar irradiance data
- Copernicus DEM for terrain models
- OpenStreetMap for parcel geometries
- Streamlit for rapid dashboard development