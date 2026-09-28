# ============================================================
# feature_target.py
# FEATURE - TARGET SEPARATION
# ============================================================

import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/placement_predict_cleaned.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "PlacementStatus"


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET])

y = df[TARGET]


# ============================================================
# 4. REMOVE NON-PREDICTIVE ID COLUMN
# ============================================================

if "StudentID" in X.columns:
    X = X.drop(columns=["StudentID"])


# ============================================================
# 5. DISPLAY INFORMATION
# ============================================================

print("========================================")
print("FEATURE - TARGET SEPARATION")
print("========================================")

print("\nDataset Shape:", df.shape)

print("\nFeature Shape:", X.shape)

print("Target Shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(TARGET)

print("\nTarget Distribution:")
print(y.value_counts())