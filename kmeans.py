# ============================================================
# kmeans.py
# K-MEANS CLUSTERING
# ============================================================

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from feature_target import X


# ============================================================
# 1. SELECT NUMERICAL FEATURES
# ============================================================

X_numeric = X.select_dtypes(
    include=["int64", "float64"]
)


# ============================================================
# 2. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_numeric
)


# ============================================================
# 3. CREATE K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# ============================================================
# 4. FIT MODEL
# ============================================================

clusters = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 5. ADD CLUSTER LABELS
# ============================================================

X_clustered = X_numeric.copy()

X_clustered["Cluster"] = clusters


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("========================================")
print("K-MEANS CLUSTERING")
print("========================================")

print("\nNumber of Clusters:", 3)

print("\nCluster Counts:")
print(X_clustered["Cluster"].value_counts())

print("\nInertia:")
print(kmeans.inertia_)


# ============================================================
# 7. VISUALIZATION
# ============================================================

if X_scaled.shape[1] >= 2:

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters
    )

    plt.xlabel(X_numeric.columns[0])
    plt.ylabel(X_numeric.columns[1])

    plt.title("K-Means Clustering")

    plt.show()