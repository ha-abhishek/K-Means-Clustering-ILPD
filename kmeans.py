import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data = pd.read_csv("ILPD.csv")

# Separate target and feature variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data columsn included in the dataset
# selected_cols = ['Age/', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
selected_cols = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "sgot", "TP", "ALB", "AGRatio", "Target"]
data = data.iloc[:, :-1]

# Standardizing the data by computing mean and standard deviation
training_mean = np.mean(X_train, axis=0)
training_std = np.std(X_train, axis=0)

X_train_std = (X_train - training_mean) / training_std
X_test_std = (X_test - training_mean) / training_std


data_array = data.values
print(data_array)

# Initializing the centroids in random
def initiliaze_centroid(k, data):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids


# Assigning the datapoints to teh cluster it is closest to
def assign_to_clusters(data, centroids):
    distance = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distance, axis=0)
    return clusters

# Update the center of each cluster by computing the mean of all data instances
def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            new_centroids[i] = data[np.random.choice(data.shape[0], 1)]
    return new_centroids


# Calling the kmeans clustering to determine the number of clusters
def kmeans(data, k, max_iter=100):
    centroids = initiliaze_centroid(k, data)
    for _ in range(max_iter):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Computing Silhouette scores for silhouette analysis for each value of k
def calculate_silhouette_score(data, clusters):
    n_samples = len(data)
    silhouette_scores = np.zeros(n_samples)
    for i in range(n_samples):
        cluster_i = clusters[i]
        cluster_i_data = data[clusters == cluster_i]
        a_i = np.mean(np.linalg.norm(data[i] - cluster_i_data, axis=1))

        b_i = np.inf
        for j in range(len(np.unique(clusters))):
            if j != cluster_i:
                cluster_j_data = data[clusters == j]
                b_ij = np.mean(np.linalg.norm(data[i] - cluster_j_data, axis=1))
                b_i = min(b_i, b_ij)

        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

    avg_silhouette_score = np.mean(silhouette_scores)
    return avg_silhouette_score


# Experiment with multiple K values
K_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
converged_epochs = []
WCSS_values = []
silhouette_scores = []

for k in K_values:
    centroids, clusters = kmeans(data_array, k)
    converged_epochs.append(len(clusters))

    # Compute Within-Cluster Sum of Squares for Elbow method
    wcss = 0
    for i in range(k):
        cluster_points = data_array[clusters == i]
        wcss += np.sum((cluster_points - centroids[i]) ** 2)
    WCSS_values.append(wcss)

    silhouette_scores.append(calculate_silhouette_score(data_array, clusters))


# Plotting for Elbow method and Silhouette Scores
pyplot.figure(figsize=(12, 6))

# Elbow Method - WCSS vs K
pyplot.subplot(1, 2, 1)
pyplot.plot(K_values, WCSS_values, marker='o')
pyplot.xlabel('Number of Clusters (K)')
pyplot.ylabel('Within-Cluster Sum of Squares')
pyplot.title('Elbow Method for Optimal K')
pyplot.grid(True)

# Silhouette Analysis - Silhouette Score vs K
pyplot.subplot(1, 2, 2)
pyplot.plot(K_values, silhouette_scores, marker='o')
pyplot.xlabel('Number of Clusters (K)')
pyplot.ylabel('Silhouette Score')
pyplot.title('Silhouette Analysis for Optimal K')
pyplot.grid(True)

pyplot.tight_layout()
pyplot.show()

# Print the required values
print("Number of Epochs to Converge for different values of k", converged_epochs)
print("Optimal K based on Elbow Method:", K_values[np.argmin(WCSS_values)])
print("Optimal K based on Silhouette Analysis:", K_values[np.argmax(silhouette_scores)])

# PCA for Optimal K based on Silhouette Analysis
optimal_k = K_values[np.argmax(silhouette_scores)]
print('optimal k value', optimal_k)
centroids, clusters = kmeans(data_array, optimal_k)

pca= PCA(2)
data_pca = pca.fit_transform(data_array)

df_pca = pd.DataFrame(data_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = clusters

pyplot.figure(figsize=(8, 6))
for k in range(optimal_k):
    pyplot.scatter(df_pca[df_pca["Cluster"] == k]["PC1"],
                   df_pca[df_pca["Cluster"] == k]["PC2"],
                   label = f'Cluster{k}')

centroids_pca = pca.transform(centroids)
pyplot.scatter(centroids_pca[:,0], centroids_pca[:, 1], c="black", marker="X", s=200, label="Centroids")

pyplot.xlabel('Principal Component 1')
pyplot.xlabel('Principal Component 2')
pyplot.title('Cluster Visualization using PCA')
pyplot.legend()
pyplot.grid()
pyplot.show()

# PCA for Optimal K based on Elbow Method
optimal_k = K_values[np.argmin(WCSS_values)]
print('optimal k value', optimal_k)
centroids, clusters = kmeans(data_array, optimal_k)

pca= PCA(2)
data_pca = pca.fit_transform(data_array)

df_pca = pd.DataFrame(data_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = clusters

pyplot.figure(figsize=(8, 6))
for k in range(optimal_k):
    pyplot.scatter(df_pca[df_pca["Cluster"] == k]["PC1"],
                   df_pca[df_pca["Cluster"] == k]["PC2"],
                   label = f'Cluster{k}')

centroids_pca = pca.transform(centroids)
pyplot.scatter(centroids_pca[:,0], centroids_pca[:, 1], c="black", marker="X", s=200, label="Centroids")

pyplot.xlabel('Principal Component 1')
pyplot.xlabel('Principal Component 2')
pyplot.title('Cluster Visualization using PCA')
pyplot.legend()
pyplot.grid()
pyplot.show()