# ============================================================
# optimization2.py
# PART 2 — BAYESIAN OPTIMIZATION + LEARNING CURVES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

from train_model1 import X_train, y_train, X_val, y_val
from optimization1 import best_model


# ============================================================
# 1. BAYESIAN-STYLE PARAMETER SEARCH
# ============================================================

print("\n========================================")
print("BAYESIAN OPTIMIZATION")
print("========================================")

# Candidate parameter combinations
candidate_params = [
    {"model__n_estimators": 100, "model__max_depth": 10},
    {"model__n_estimators": 200, "model__max_depth": 10},
    {"model__n_estimators": 200, "model__max_depth": 20},
    {"model__n_estimators": 300, "model__max_depth": 20},
    {"model__n_estimators": 300, "model__max_depth": 30}
]

best_score = -np.inf
best_params = None
bayesian_model = None


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
        bayesian_model = model


print("\nBest Parameters:")
print(best_params)

print("Best Validation Accuracy:")
print(round(best_score, 4))


# ============================================================
# 2. LEARNING CURVE
# ============================================================

print("\n========================================")
print("LEARNING CURVE")
print("========================================")


train_sizes, train_scores, validation_scores = learning_curve(
    bayesian_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)


# Calculate mean scores

train_mean = train_scores.mean(axis=1)

validation_mean = validation_scores.mean(axis=1)


# ============================================================
# 3. PLOT LEARNING CURVE
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
# 4. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("OPTIMIZATION COMPLETED")
print("========================================")