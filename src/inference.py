import pandas as pd
import lightgbm as lgb

df = pd.read_parquet("data/parcels_features.parquet")
X = df[["NSRDB_GHI_mean", "mean_slope"]]

model = lgb.Booster(model_file="models/lgb_site_model.txt")
df["pred_energy"] = model.predict(X)
df["suitability_score"] = (df["pred_energy"] - df["pred_energy"].min()) / (
    df["pred_energy"].max() - df["pred_energy"].min()
)

df.to_csv("data/scored_parcels.csv", index=False)
print("Saved data/scored_parcels.csv")
