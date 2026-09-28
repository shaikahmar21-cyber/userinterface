# ============================================================
# train_model3.py
# PART 3 — MODEL SAVING
# ============================================================

import os
import joblib

from train_model2 import (
    logistic_pipeline,
    random_forest_pipeline,
    gradient_boosting_pipeline
)


# ============================================================
# 1. MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. SAVE LOGISTIC REGRESSION
# ============================================================

joblib.dump(
    logistic_pipeline,
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

print("Logistic Regression model saved.")


# ============================================================
# 3. SAVE RANDOM FOREST
# ============================================================

joblib.dump(
    random_forest_pipeline,
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

print("Random Forest model saved.")


# ============================================================
# 4. SAVE GRADIENT BOOSTING
# ============================================================

joblib.dump(
    gradient_boosting_pipeline,
    os.path.join(MODEL_DIR, "gradient_boosting.pkl")
)

print("Gradient Boosting model saved.")


# ============================================================
# 5. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("All trained models saved successfully!")
print("========================================")