import os
import sys
import io
import contextlib
import pandas as pd
import numpy as np
import yaml
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ===============================
# PATHS & SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "parcels_features.parquet")
RESULTS_PATH = os.path.join(BASE_DIR, "..", "data", "model_comparison.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config.yaml")
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "model_optimize.log")

# ===============================
# SILENT EXECUTION CONTEXT
# ===============================
import warnings
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_output():
    """Context manager to silence stdout and stderr."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield

# ===============================
# LOAD DATA
# ===============================
def load_data():
    print("📂 Loading feature dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    target_col = "suitability_proxy"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(float)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# MODEL DEFINITIONS
# ===============================
def get_models():
    return {
        "LightGBM": lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_seed=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        ),
    }

# ===============================
# MAIN BENCHMARK LOGIC
# ===============================
def run_benchmark():
    X_train, X_val, y_train, y_val = load_data()
    models = get_models()

    results = []
    for name, model in models.items():
        print(f"\n⚙️ Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        print(f"✅ {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # Save results
    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(RESULTS_PATH, index=False)
    best_model = results_df.iloc[0]["Model"]

    print(f"\n🏁 Model comparison complete! Best model → {best_model}")
    print(f"📊 Results saved to: {RESULTS_PATH}")

    # Update config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config["model"] = best_model.lower()

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

    print(f"✅ Updated config.yaml with best model: {best_model}")
    return best_model, results_df


# ===============================
# ENTRY POINT (Silent Mode)
# ===============================
if __name__ == "__main__":
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        best_model, results_df = run_benchmark()

    # Save all details to a log file
    with open(LOG_FILE, "w") as logf:
        logf.write(buffer.getvalue())
