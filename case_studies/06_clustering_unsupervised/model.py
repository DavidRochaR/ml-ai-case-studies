"""Clustering — model definitions."""
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA


def get_clusterers(n_clusters=4):
    return {
        "K-Means": KMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }


def reduce_dimensions(X, n_components=2):
    return PCA(n_components=n_components, random_state=42).fit_transform(X)
