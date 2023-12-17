import pandas as pd

from models.Similarity import Similarity

class Clustering(Similarity):
    def __init__(self):
        pass

    def process_input(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Process input data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        
        Returns
        -------
        data : pd.DataFrame
            Processed data.
        """
        self.input = self.__process_input(data=data)

        if method in self.available_similarities:
            self.input = self.compute_similarity(data=data, method=method)
        else:
            raise ValueError(f"Invalid similarity method: {method}.")
        
        return data
    
    def compute_clusters(self, k_opt: int) -> pd.DataFrame:
        """
        Compute clusters based on similarity matrix and optimal number of clusters.
        
        Parameters
        ----------
        similarity : pd.DataFrame
            Similarity matrix.
        k_opt : int
            Optimal number of clusters.
        
        Returns
        -------
        clusters : pd.DataFrame
            Clusters.
        """
        pass