import pandas as pd
from tqdm import tqdm

from models.Similarity import Similarity

def run_memory(data: pd.DataFrame,
               fix_start: bool,
               estimation_window: int,
               similarity_method: str,
               k_opt_method: str,
               clustering_method: str) -> dict:
    
    simi = Similarity(method=similarity_method)
    
    for step in tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window, desc="Build Memory"):

        if fix_start or (step == 0):
            start = 0
        else:
            start += 1

        # subset train data
        train = data.iloc[start:(estimation_window + step), :]

        # compute similarity measure
        similarity = simi.compute_similarity(train=train, method=clustering_method)

        # compute optimal number of clusters
        k_opt = simi.compute_k_opt(similarity=similarity, method=k_opt_method)

        # compute easy clusters
        clusters = simi.compute_clusters(similarity=similarity, k_opt=3)

        # compute hard clusters
        clusters = simi.compute_clusters(similarity=similarity, k_opt=k_opt)

        # compute transition probabilities
        transition_probs = simi.compute_transition_probs(clusters=clusters)
        

