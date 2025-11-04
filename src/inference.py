import os, pandas as pd, lightgbm as lgb
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_BIN = os.path.join(MODEL_DIR, "lgb_site_model.bin")
MODEL_TXT = os.path.join(MODEL_DIR, "lgb_site_model.txt")  # fallback

feat = pd.read_parquet(os.path.join(DATA_DIR, "parcels_features.parquet"))

# Build X the same way as in training
drop_cols = {"parcel_id", "suitability_proxy", "pred_energy", "suitability_score"}
num_df = feat.select_dtypes(include=[np.number]).copy()
X = num_df.drop(columns=[c for c in drop_cols if c in num_df.columns], errors="ignore").fillna(0)

# Fallback: one-hot landuse if training had it (not strictly required, but safe)
if X.shape[1] < 2 and "landuse" in feat.columns:
    X = pd.concat([X, pd.get_dummies(feat["landuse"].astype(str), prefix="lu")], axis=1).fillna(0)

# Load model
if os.path.exists(MODEL_BIN):
    model = lgb.Booster(model_file=MODEL_BIN)
elif os.path.exists(MODEL_TXT):
    model = lgb.Booster(model_file=MODEL_TXT)
else:
    raise FileNotFoundError("No LightGBM model found. Train with `python3 src/model_train.py`.")

# Predict
pred = model.predict(X, num_iteration=model.best_iteration)
feat["pred_energy"] = pred

# Normalized suitability 0..1 (safe if constant)
pmin, pmax = float(pred.min()), float(pred.max())
den = (pmax - pmin) if pmax > pmin else 1.0
feat["suitability_score"] = (pred - pmin) / den

# Save compact CSV for the app
cols = ["parcel_id", "NSRDB_GHI_mean", "mean_slope", "pred_energy", "suitability_score"]
for c in cols:
    if c not in feat.columns:
        feat[c] = 0
feat[cols].to_csv(os.path.join(DATA_DIR, "scored_parcels.csv"), index=False)
print("✅ Predictions saved → data/scored_parcels.csv")
