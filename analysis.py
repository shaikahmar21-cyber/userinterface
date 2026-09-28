# ============================================================
# optimization.py
# HYPERPARAMETER OPTIMIZATION + LEARNING CURVE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)
from sklearn.metrics import accuracy_score

from train_model1 import (
    X_train,
    y_train,
    X_val,
    y_val
)

from train_model1 import random_forest_pipeline


# ============================================================
# 1. GRID SEARCH
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
# 2. RANDOMIZED SEARCH
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
# 3. SELECT BEST MODEL
# ============================================================

best_model = random_search.best_estimator_


# ============================================================
# 4. PARAMETER SEARCH
# ============================================================

print("\n========================================")
print("PARAMETER OPTIMIZATION")
print("========================================")

candidate_params = [
    {
        "model__n_estimators": 100,
        "model__max_depth": 10
    },
    {
        "model__n_estimators": 200,
        "model__max_depth": 10
    },
    {
        "model__n_estimators": 200,
        "model__max_depth": 20
    },
    {
        "model__n_estimators": 300,
        "model__max_depth": 20
    },
    {
        "model__n_estimators": 300,
        "model__max_depth": 30
    }
]

best_score = -np.inf
best_params = None
optimized_model = None


for params in candidate_params:

    model = best_model.set_params(**params)

    model.fit(X_train, y_train)

    validation_pred = model.predict(X_val)

    score = accuracy_score(
        y_val,
        validation_pred
    )

    print(
        f"Parameters: {params} | "
        f"Validation Accuracy: {score:.4f}"
    )

    if score > best_score:
        best_score = score
        best_params = params
        optimized_model = model


print("\nBest Parameters:")
print(best_params)

print("\nBest Validation Accuracy:")
print(round(best_score, 4))


# ============================================================
# 5. LEARNING CURVE
# ============================================================

print("\n========================================")
print("LEARNING CURVE")
print("========================================")

train_sizes, train_scores, validation_scores = learning_curve(
    optimized_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)


# ============================================================
# 6. CALCULATE MEAN SCORES
# ============================================================

train_mean = train_scores.mean(axis=1)

validation_mean = validation_scores.mean(axis=1)


# ============================================================
# 7. PLOT LEARNING CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")

plt.title("Learning Curve")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 8. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("OPTIMIZATION COMPLETED")
print("========================================")