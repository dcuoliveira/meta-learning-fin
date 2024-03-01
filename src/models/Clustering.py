import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from models.Similarity import Similarity

class Clustering(Similarity):
    def __init__(self, similarity_method: str=None, max_k: int=10, max_iter: int=800, n_init: int=1):
        self.similarity_method = similarity_method
        self.max_k = max_k
        self.max_iter = max_iter
        self.n_init = n_init

    def elbow(self, data: np.array):
        """
        Compute optimal number of clusters using elbow method.

        Parameters
        ----------
        data : np.array
            Input data.

        Returns
        -------
        k_opt : int
            Optimal number of clusters.
        """

        # using the Elbow method to find the optimal number of clusters
        wcss = []  # Within-cluster sum of squares
        for i in range(1, self.max_k+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=self.max_iter, n_init=self.n_init, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        # determine the optimal number of clusters using KneeLocator
        kn = KneeLocator(range(1, self.max_k+1), wcss, curve='convex', direction='decreasing')
        optimal_k = kn.knee

        return optimal_k
    
    def compute_k_opt(self, data: np.array, k_opt_method: str):
        """
        Compute optimal number of clusters.

        Parameters
        ----------
        data : np.array
            Input data.
        k_opt_method : str
            Method to compute optimal number of clusters.

        Returns
        -------
        k_opt : int
            Optimal number of clusters.
        """
        
        if k_opt_method == "elbow":
            return self.elbow(data=data)
        else:
            raise ValueError("Invalid k_opt_method.")
        
    def kmeans_wrapper(self, data: np.array, k: int, extra_data: np.array = None):
        """
        Wrapper for sklearn.cluster.KMeans.

        Parameters
        ----------
        data : np.array
            Input data.
        k : int
            Number of clusters.
        
        Returns
        -------
        clusters : np.array
            Clusters.
        """

        # kmeans method
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=self.max_iter, n_init=self.n_init, random_state=0)
        kmeans.fit(data)
        clusters = kmeans.predict(data)
        centroids = kmeans.cluster_centers_
        probs = euclidean_distances(extra_data, centroids)
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs = 1 - probs
        probs = probs / probs.sum(axis=1, keepdims=True)

        # gaussian mixture method
        #gmm = GaussianMixture(n_components=k, max_iter=self.max_iter, n_init=self.n_init, random_state=0, init_params='k-means++')
        #gmm.fit(data)
        #clusters = gmm.predict(data)
        #centroids = gmm.means_
        #probs = gmm.predict_proba(extra_data)

        return clusters, centroids, probs
    
    def compute_clusters(self,
                         data: np.array,
                         extra_data: np.array=None,
                         method: str="kmeans",
                         k: int=None,
                         k_opt_method: str="elbow",
                         similarity_method: str=None) -> pd.DataFrame:
        """
        Compute clusters based on data and optimal number of clusters.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        method : str, optional
            Clustering method, by default "kmeans".
        k : int, optional
            Number of clusters, by default None.
        k_opt_method : str, optional
            Method to compute optimal number of clusters, by default "elbow".
        similarity_method : str, optional
            Similarity method, by default None.
        
        Returns
        -------
        clusters : pd.DataFrame
            Clusters.
        """

        if similarity_method is not None:
            self.similarity_method = similarity_method

        self.orig_data = data.copy()

        # preprocess data
        if self.similarity_method is not None:
            data, extra_data = self.compute_similarity(data=data, method=self.similarity_method, extra_data=extra_data)
        
        # check if any default k is provided    
        if k is None:
            k = self.compute_k_opt(data=data, k_opt_method=k_opt_method)

        # compute clusters
        if method == "kmeans":
            clusters, centroids, probs = self.kmeans_wrapper(data=data, k=k, extra_data=extra_data)
        else:
            raise ValueError("Invalid clustering method.")
        
        return clusters, centroids, probs
