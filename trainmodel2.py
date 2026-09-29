

import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from preprocess import load_and_prepare_data


# ============================================================
# 1. MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

X, y, preprocessor = load_and_prepare_data()


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% development data, 20% test data
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second: 75% of development data for training,
#         25% for validation
# Final ratio = 60% train, 20% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)


# ============================================================
# 4. CREATE MODEL PIPELINES
# ============================================================

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


gradient_boosting_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# ============================================================
# 5. DISPLAY DATASET SPLIT
# ============================================================

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)

print("\nTrain_model1 setup completed.")