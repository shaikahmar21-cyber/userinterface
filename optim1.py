# ============================================================
# optimization1.py
# PART 1 — HYPERPARAMETER SEARCH
# ============================================================

import numpy as np

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

from train_model1 import X_train, y_train
from train_model3 import random_forest_pipeline


# ============================================================
# 1. GRID SEARCH — RANDOM FOREST
# ============================================================

print("\n========================================")
print("GRID SEARCH")
print("========================================")

grid_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}


grid_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=grid_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


print("\nBest Grid Search Parameters:")
print(grid_search.best_params_)

print("\nBest Grid Search Score:")
print(round(grid_search.best_score_, 4))


# ============================================================
# 2. RANDOMIZED SEARCH — RANDOM FOREST
# ============================================================

print("\n========================================")
print("RANDOMIZED SEARCH")
print("========================================")

random_params = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}


random_search = RandomizedSearchCV(
    estimator=random_forest_pipeline,
    param_distributions=random_params,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)


print("\nBest Randomized Search Parameters:")
print(random_search.best_params_)

print("\nBest Randomized Search Score:")
print(round(random_search.best_score_, 4))


# ============================================================
# 3. BEST MODEL
# ============================================================

best_model = random_search.best_estimator_

print("\n========================================")
print("OPTIMIZATION PART 1 COMPLETED")
print("========================================")