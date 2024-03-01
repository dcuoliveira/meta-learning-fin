import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

class Similarity:
    def __init__(self, item1, item2, similarity):
        self.available_similarities = ["cosine_similarity", "euclidean_distance", "manhattan_distance"]

    def compute_similarity(self, data: np.array, method: str, extra_data: np.array = None) -> pd.DataFrame:
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
        extra_similarity = None
        if method == "cosine":
            similarity = cosine_similarity(data)
            if extra_data is not None:
                extra_similarity = cosine_similarity(extra_data, data)
        elif method == "euclidean":
            similarity = euclidean_distances(data)
            if extra_data is not None:
                extra_similarity = euclidean_distances(extra_data, data)
        elif method == "manhattan":
            similarity = manhattan_distances(data)
            if extra_data is not None:
                extra_similarity = manhattan_distances(extra_data, data)
        else:
            raise ValueError(f"Invalid similarity method: {method}.")

        return similarity, extra_similarity