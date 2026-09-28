# ============================================================
# evaluate_model1.py
# PART 1 — LOAD MODELS AND GENERATE PREDICTIONS
# ============================================================

import os
import joblib

from train_model1 import X_test, y_test


# ============================================================
# 1. MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

logistic_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

gradient_boosting_model = joblib.load(
    os.path.join(MODEL_DIR, "gradient_boosting.pkl")
)


# ============================================================
# 3. GENERATE TEST PREDICTIONS
# ============================================================

logistic_pred = logistic_model.predict(X_test)

random_forest_pred = random_forest_model.predict(X_test)

gradient_boosting_pred = gradient_boosting_model.predict(X_test)


# ============================================================
# 4. GENERATE PROBABILITY PREDICTIONS
# ============================================================

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

random_forest_prob = random_forest_model.predict_proba(X_test)[:, 1]

gradient_boosting_prob = gradient_boosting_model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. DISPLAY COMPLETION
# ============================================================

print("========================================")
print("Models loaded successfully.")
print("Test predictions generated.")
print("========================================")

print("\nTest samples:", len(X_test))