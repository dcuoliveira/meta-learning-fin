import pandas as pd
from tqdm import tqdm
import warnings

from models.Clustering import Clustering

warnings.filterwarnings("ignore")

def run_memory(data: pd.DataFrame,
               fix_start: bool,
               estimation_window: int,
               similarity_method: str,
               clustering_method: str,
               k_opt_method: str) -> dict:
    
    clustering = Clustering(similarity_method=similarity_method)
    
    pbar = tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window)
    all_clusters = []
    for step in pbar:

        if fix_start or (step == 0):
            start = 0
        else:
            start += 1

        # subset train data
        train = data.iloc[start:(estimation_window + step), :]

        # compute clusters for easy days
        clusters = clustering.compute_clusters(data=train, method=clustering_method, k=3, k_opt_method=k_opt_method)

        # subset clusters that appear less frequently
        ## add clusters to train data
        train["cluster"] = clusters

        ## compute cluster frequencies
        cluster_freq = train["cluster"].value_counts(normalize=True)

        ## subset clusters that appear less frequently
        min_cluster = cluster_freq[cluster_freq == cluster_freq.min()].index[0]

        ## subset train data
        train_easy = train[train["cluster"] == min_cluster]
        train_hard = train[train["cluster"] != min_cluster]

        # compute clusters for hard days
        clusters = clustering.compute_clusters(data=train_hard, method=clustering_method, k_opt_method=k_opt_method)
        train_hard["cluster"] = clusters
        train_easy["cluster"] = clusters.max() + 1

        # merge easy and hard clusters
        train = pd.concat([train_easy, train_hard]).sort_index()

        all_clusters.append(train[["cluster"]].rename(columns={"cluster": f"cluster_step{step}"}))
        pbar.set_description(f"Building memory using window: {step}")

    all_clusters_df = pd.concat(all_clusters, axis=1)

    return all_clusters_df

        

