# ============================================================
# evaluate_model2.py
# PART 2 — CLASSIFICATION METRICS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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
# 1. LOGISTIC REGRESSION METRICS
# ============================================================

logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)
logistic_f1 = f1_score(y_test, logistic_pred)
logistic_auc = roc_auc_score(y_test, logistic_prob)


# ============================================================
# 2. RANDOM FOREST METRICS
# ============================================================

rf_accuracy = accuracy_score(y_test, random_forest_pred)
rf_precision = precision_score(y_test, random_forest_pred)
rf_recall = recall_score(y_test, random_forest_pred)
rf_f1 = f1_score(y_test, random_forest_pred)
rf_auc = roc_auc_score(y_test, random_forest_prob)


# ============================================================
# 3. GRADIENT BOOSTING METRICS
# ============================================================

gb_accuracy = accuracy_score(y_test, gradient_boosting_pred)
gb_precision = precision_score(y_test, gradient_boosting_pred)
gb_recall = recall_score(y_test, gradient_boosting_pred)
gb_f1 = f1_score(y_test, gradient_boosting_pred)
gb_auc = roc_auc_score(y_test, gradient_boosting_prob)


# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MODEL EVALUATION RESULTS")
print("========================================")

print("\nLogistic Regression")
print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1-Score :", round(logistic_f1, 4))
print("ROC-AUC  :", round(logistic_auc, 4))


print("\nRandom Forest")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1-Score :", round(rf_f1, 4))
print("ROC-AUC  :", round(rf_auc, 4))


print("\nGradient Boosting")
print("Accuracy :", round(gb_accuracy, 4))
print("Precision:", round(gb_precision, 4))
print("Recall   :", round(gb_recall, 4))
print("F1-Score :", round(gb_f1, 4))
print("ROC-AUC  :", round(gb_auc, 4))