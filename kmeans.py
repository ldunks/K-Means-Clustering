# Assignmnet 2
# Liam Duncan - 301476562
# March 19, 2024

import numpy as np

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])

        while iteration < self.max_iter:
            for i in range(X.shape[0]):
                distances = self.euclidean_distance(X[i], self.centroids)
                clustering[i] = int(np.argmin(distances))
            self.update_centroids(clustering, X)
            iteration += 1

        print(self.silhouette(clustering,X))
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):

        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for cluster_idx in range(self.n_clusters):

            cluster_points = X[clustering == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = (np.mean(cluster_points, axis=0))
            else:
                new_centroids[cluster_idx] = (X[np.random.randint(X.shape[0])])
        
        self.centroids = new_centroids

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        np.random.seed(23)
        if self.init == 'random':
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        elif self.init == 'kmeans++':

            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            self.centroids[0] = X[np.random.choice(X.shape[0])]

            for i in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(point - centroid)**2 for centroid in self.centroids]) for point in X])
                probabilities = distances / np.sum(distances)
                next_centroid_idx = np.random.choice(X.shape[0], p=probabilities)
                self.centroids[i] = X[next_centroid_idx]

        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        squared_diff = np.sum((X1 - X2) ** 2, axis=1)
        dist = np.sqrt(squared_diff)
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        silhouette_vals = []
        for i in range(len(clustering)):
            cluster_i = int(clustering[i])
            dist_a = np.sqrt(np.sum((X[i] - self.centroids[cluster_i]) ** 2))
            
            dist_b = np.inf
            for j in range(len(self.centroids)):
                if j != cluster_i:
                    dist_temp = np.sqrt(np.sum((X[i] - self.centroids[j]) ** 2))
                    if dist_temp < dist_b:
                        dist_b = dist_temp

            s_i = (dist_b - dist_a) / max (dist_b, dist_a)
            silhouette_vals.append(s_i)

        silhouette_avg = np.mean(silhouette_vals)
        return silhouette_avg

