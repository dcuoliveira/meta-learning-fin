import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

class Similarity:
    def __init__(self, item1, item2, similarity):
        self.available_similarities = ["cosine_similarity", "euclidean_distance", "manhattan_distance"]

    def compute_similarity(self, data: np.array, method: str) -> pd.DataFrame:
        """
        Compute similarity matrix.
        
        Parameters
        ----------
        method : str
            Similarity method.
        
        Returns
        -------
        similarity : pd.DataFrame
            Similarity matrix.
        """
        if method == "cosine_similarity":
            similarity = cosine_similarity(data)
        elif method == "euclidean_distance":
            similarity = euclidean_distances(data)
        elif method == "manhattan_distance":
            similarity = manhattan_distances(data)
        else:
            raise ValueError(f"Invalid similarity method: {method}.")

        return similarity