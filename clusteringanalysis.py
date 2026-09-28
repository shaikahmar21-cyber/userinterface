# ============================================================
# clustering_analysis.py
# CLUSTERING ANALYSIS
# ============================================================

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from feature_target import X


# ============================================================
# 1. SELECT NUMERICAL FEATURES
# ============================================================

X_numeric = X.select_dtypes(
    include=["int64", "float64"]
)


# ============================================================
# 2. SCALE DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_numeric
)


# ============================================================
# 3. K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 4. MINI-BATCH K-MEANS
# ============================================================

mini_kmeans = MiniBatchKMeans(
    n_clusters=3,
    batch_size=32,
    random_state=42,
    n_init=10
)

mini_labels = mini_kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 5. SILHOUETTE SCORES
# ============================================================

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

mini_score = silhouette_score(
    X_scaled,
    mini_labels
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("========================================")
print("CLUSTERING ANALYSIS")
print("========================================")

print("\nK-Means")
print("Inertia:", round(kmeans.inertia_, 4))
print("Silhouette Score:", round(kmeans_score, 4))

print("\nMini-Batch K-Means")
print("Inertia:", round(mini_kmeans.inertia_, 4))
print("Silhouette Score:", round(mini_score, 4))

print("\n========================================")
print("ANALYSIS COMPLETED")
print("========================================")