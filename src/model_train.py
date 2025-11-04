import os, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

try:
    import optuna
except ImportError:
    optuna = None
    warnings.warn("Optuna not installed. Will train with a strong default config.")

# -------------------------
# Paths & I/O
# -------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEAT_PATH  = os.path.join(DATA_DIR, "parcels_features.parquet")
MODEL_BIN  = os.path.join(MODEL_DIR, "lgb_site_model.bin")   # binary (faster load)
MODEL_TXT  = os.path.join(MODEL_DIR, "lgb_site_model.txt")   # optional text copy
PARAMS_OUT = os.path.join(MODEL_DIR, "best_params.json")
FIMP_PATH  = os.path.join(MODEL_DIR, "feature_importance.csv")

# -------------------------
# Load data
# -------------------------
df = pd.read_parquet(FEAT_PATH)

# Ensure a stable target. Prefer existing proxy; else build one.
if "suitability_proxy" in df.columns:
    y = df["suitability_proxy"].copy()
else:
    ghi = df.get("NSRDB_GHI_mean", pd.Series(0, index=df.index))
    slope = df.get("mean_slope", pd.Series(0, index=df.index)).replace({0: 1e-6})
    y = (ghi / (1 + slope)).fillna(0)

# Build feature matrix:
# - Keep numeric columns only
# - Drop obvious non-features
drop_cols = {
    "parcel_id", "suitability_proxy",            # IDs / target
    "pred_energy", "suitability_score",          # downstream preds
}

num_df = df.select_dtypes(include=[np.number]).copy()
X = num_df.drop(columns=[c for c in drop_cols if c in num_df.columns], errors="ignore").fillna(0)

# Safety: ensure at least 2 features
if X.shape[1] < 2:
    # try add one-hot landuse if numeric features are too few
    if "landuse" in df.columns:
        landuse_ohe = pd.get_dummies(df["landuse"].astype(str), prefix="lu")
        X = pd.concat([X, landuse_ohe], axis=1)
    # final guard
    if X.shape[1] < 2:
        raise ValueError(f"Not enough features to train. Got shape={X.shape}. "
                         f"Make sure features.py is producing more numeric features.")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
dval   = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False)

# -------------------------
# Base params (fast & strong)
# GPU toggle: set env LGB_DEVICE=gpu to try GPU, else CPU
# -------------------------
device_type = "gpu" if os.environ.get("LGB_DEVICE", "").lower() == "gpu" else "cpu"

base_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "num_threads": -1,
    "verbosity": -1,
}

# LightGBM device param (works across versions)
if device_type == "gpu":
    base_params.update({"device_type": "gpu"})
else:
    base_params.update({"device_type": "cpu"})

# -------------------------
# Optuna tuning (optional but recommended)
# -------------------------
def train_eval(params):
    callbacks = [lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(20)]
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=600,
        callbacks=callbacks
    )
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = sqrt(mean_squared_error(y_val, preds))
    return model, rmse

best_params = base_params.copy()
best_rmse = float("inf")

if optuna is not None:
    def objective(trial):
        params = base_params.copy()
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        })
        model, rmse = train_eval(params)
        return rmse

    n_trials = int(os.environ.get("OPTUNA_TRIALS", "30"))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params.update(study.best_params)

# -------------------------
# Final train with best params
# -------------------------
final_model, final_rmse = train_eval(best_params)

# Save artifacts
with open(PARAMS_OUT, "w") as f:
    json.dump({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device_type": device_type,
        "best_params": best_params,
        "val_rmse": final_rmse,
        "n_features": X.shape[1],
    }, f, indent=2)

# Save model in binary (fast) and text (human-readable) forms
# --- Safe Model Save (Cross-version) ---
if hasattr(final_model, "save_model"):
    try:
        # LightGBM & CatBoost
        save_args = {}
        if "format" in final_model.save_model.__code__.co_varnames:
            save_args["format"] = "binary"
        if "num_iteration" in final_model.save_model.__code__.co_varnames:
            save_args["num_iteration"] = getattr(final_model, "best_iteration", None)
        final_model.save_model(MODEL_BIN, **save_args)
    except Exception as e:
        print(f"[WARN] save_model() failed with args, retrying basic save: {e}")
        try:
            final_model.save_model(MODEL_BIN)
        except Exception:
            import joblib
            joblib.dump(final_model, MODEL_BIN.replace(".bin", ".pkl"))
else:
    # sklearn-style models (RandomForest, Ridge, etc.)
    import joblib
    joblib.dump(final_model, MODEL_BIN.replace(".bin", ".pkl"))
print(f"✅ Model saved: {MODEL_BIN}")

final_model.save_model(MODEL_TXT, num_iteration=final_model.best_iteration)

# Feature importance
fimp = pd.DataFrame({
    "feature": X.columns,
    "gain_importance": final_model.feature_importance(importance_type="gain"),
    "split_importance": final_model.feature_importance(importance_type="split"),
}).sort_values("gain_importance", ascending=False)
fimp.to_csv(FIMP_PATH, index=False)

print(f"✅ Model saved (binary): {MODEL_BIN}")
print(f"✅ Model saved (text)  : {MODEL_TXT}")
print(f"✅ Best params saved   : {PARAMS_OUT}")
print(f"✅ Feature importances : {FIMP_PATH}")
print(f"Validation RMSE: {final_rmse:.4f}")

# -------------------------
# Optional: SHAP explainability
# -------------------------
try:
    import shap, matplotlib.pyplot as plt
    shap.summary_plot(
        shap.TreeExplainer(final_model).shap_values(X_val, y_val, check_additivity=False),
        X_val, show=False
    )
    plt.tight_layout()
    shap_path = os.path.join(MODEL_DIR, "shap_summary.png")
    plt.savefig(shap_path, dpi=160)
    plt.close()
    print(f"✅ SHAP summary saved  : {shap_path}")
except Exception as e:
    warnings.warn(f"SHAP summary skipped: {e}")
