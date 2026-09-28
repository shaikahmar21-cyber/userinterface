# ============================================================
# evaluate_model3.py
# PART 3 — CONFUSION MATRIX AND ROC CURVES
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from evaluate_model1 import (
    y_test,
    logistic_pred,
    random_forest_pred,
    gradient_boosting_pred,
    logistic_prob,
    random_forest_prob,
    gradient_boosting_prob
)


# ============================================================
# 1. CONFUSION MATRIX — LOGISTIC REGRESSION
# ============================================================

cm_logistic = confusion_matrix(y_test, logistic_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_logistic
)

disp.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.show()


# ============================================================
# 2. CONFUSION MATRIX — RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(y_test, random_forest_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf
)

disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()


# ============================================================
# 3. CONFUSION MATRIX — GRADIENT BOOSTING
# ============================================================

cm_gb = confusion_matrix(y_test, gradient_boosting_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_gb
)

disp.plot()
plt.title("Gradient Boosting - Confusion Matrix")
plt.show()


# ============================================================
# 4. ROC CURVES
# ============================================================

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_prob
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    random_forest_prob
)

fpr_gb, tpr_gb, _ = roc_curve(
    y_test,
    gradient_boosting_prob
)


# ============================================================
# 5. CALCULATE AUC
# ============================================================

auc_logistic = auc(
    fpr_logistic,
    tpr_logistic
)

auc_rf = auc(
    fpr_rf,
    tpr_rf
)

auc_gb = auc(
    fpr_gb,
    tpr_gb
)


# ============================================================
# 6. PLOT ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC = {auc_logistic:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.3f})"
)

plt.plot(
    fpr_gb,
    tpr_gb,
    label=f"Gradient Boosting (AUC = {auc_gb:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 7. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("Evaluation completed successfully!")
print("========================================")