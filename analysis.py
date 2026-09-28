# ============================================================
# analysis.py
# VALIDATION CURVE + McNEMAR'S TEST + PAIRED BOOTSTRAP
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from scipy.stats import chi2

from train_model1 import (
    X_train,
    y_train,
    X_test,
    y_test,
    random_forest_pipeline
)

from evaluate_model1 import (
    logistic_pred,
    random_forest_pred,
    gradient_boosting_pred
)


# ============================================================
# 1. VALIDATION CURVE
# ============================================================

print("\n========================================")
print("VALIDATION CURVE")
print("========================================")

param_range = [50, 100, 150, 200, 250, 300]

train_scores, validation_scores = validation_curve(
    random_forest_pipeline,
    X_train,
    y_train,
    param_name="model__n_estimators",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)


# ============================================================
# 2. PLOT VALIDATION CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    param_range,
    train_mean,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    param_range,
    validation_mean,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")

plt.title("Random Forest Validation Curve")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 3. McNEMAR'S TEST
# ============================================================

print("\n========================================")
print("McNEMAR'S TEST")
print("========================================")

# Compare Logistic Regression and Random Forest

correct_logistic = logistic_pred == y_test
correct_rf = random_forest_pred == y_test

# b = Logistic correct, RF wrong
b = np.sum(
    correct_logistic & ~correct_rf
)

# c = Logistic wrong, RF correct
c = np.sum(
    ~correct_logistic & correct_rf
)

print("Logistic correct / RF wrong:", b)
print("Logistic wrong / RF correct:", c)


# McNemar chi-square statistic

if b + c > 0:

    mcnemar_statistic = (
        (abs(b - c) - 1) ** 2
    ) / (b + c)

    p_value = 1 - chi2.cdf(
        mcnemar_statistic,
        df=1
    )

else:

    mcnemar_statistic = 0
    p_value = 1.0


print("McNemar Statistic:", round(mcnemar_statistic, 4))
print("p-value:", round(p_value, 4))


# ============================================================
# 4. PAIRED BOOTSTRAP
# ============================================================

print("\n========================================")
print("PAIRED BOOTSTRAP")
print("========================================")

n_iterations = 1000

rng = np.random.default_rng(42)

differences = []

n_samples = len(y_test)


for _ in range(n_iterations):

    # Sample the same test indices for both models
    indices = rng.integers(
        0,
        n_samples,
        size=n_samples
    )

    bootstrap_y = y_test.iloc[indices]

    bootstrap_logistic = logistic_pred[indices]

    bootstrap_rf = random_forest_pred[indices]

    logistic_accuracy = accuracy_score(
        bootstrap_y,
        bootstrap_logistic
    )

    rf_accuracy = accuracy_score(
        bootstrap_y,
        bootstrap_rf
    )

    differences.append(
        rf_accuracy - logistic_accuracy
    )


differences = np.array(differences)


# ============================================================
# 5. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

lower = np.percentile(
    differences,
    2.5
)

upper = np.percentile(
    differences,
    97.5
)

mean_difference = differences.mean()


print("Mean Accuracy Difference:", round(mean_difference, 4))

print("95% Confidence Interval:")
print(
    "(",
    round(lower, 4),
    ",",
    round(upper, 4),
    ")"
)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("ANALYSIS COMPLETED")
print("========================================")

print("\nValidation Curve: Completed")
print("McNemar's Test: Completed")
print("Paired Bootstrap: Completed")