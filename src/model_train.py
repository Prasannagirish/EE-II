import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap

df = pd.read_parquet("data/parcels_features.parquet")

df = df.dropna(subset=["NSRDB_GHI_mean", "mean_slope"])
df["target_energy"] = df["suitability_proxy"]

X = df[["NSRDB_GHI_mean", "mean_slope"]]
y = df["target_energy"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)

params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05, "num_leaves": 31}
model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=300, early_stopping_rounds=30)

model.save_model("models/lgb_site_model.txt")

y_pred = model.predict(X_val)
print("Validation RMSE:", mean_squared_error(y_val, y_pred, squared=False))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
