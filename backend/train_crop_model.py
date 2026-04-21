import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

# ================= LOAD DATA =================
df = pd.read_csv("Crop recommendation dataset.csv")
df.columns = df.columns.str.strip()

# ================= FEATURE ENGINEERING =================
df["N_ratio"] = df["N"] / df["N_MAX"].replace(0, 1)
df["P_ratio"] = df["P"] / df["P_MAX"].replace(0, 1)
df["K_ratio"] = df["K"] / df["K_MAX"].replace(0, 1)

# ================= FEATURE LIST =================
FEATURE_COLUMNS = [
    "N", "P", "K",
    "TEMP",
    "RELATIVE_HUMIDITY",
    "SOIL_PH",
    "MAX_TEMP",
    "WATERREQUIRED",
    "N_ratio", "P_ratio", "K_ratio"
]

X = df[FEATURE_COLUMNS]
y = df["CROPS"]

# ================= TRAIN MODEL =================
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)

# ================= SAVE MODEL =================
model_data = {
    "model": model,
    "features": FEATURE_COLUMNS
}

with open("crop_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("🔥 Advanced model trained & saved safely")

# ================= GRAPH 1: FEATURE IMPORTANCE =================
importances = model.feature_importances_

plt.figure()
plt.barh(FEATURE_COLUMNS, importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ================= GRAPH 2: FEATURE DISTRIBUTION =================
plt.figure()
df[FEATURE_COLUMNS].hist(bins=20)
plt.suptitle("Feature Distribution")
plt.tight_layout()
plt.savefig("feature_distribution.png")
plt.show()