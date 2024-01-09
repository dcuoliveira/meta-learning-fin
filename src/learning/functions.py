import pandas as pd
from tqdm import tqdm
import warnings

from models.Clustering import Clustering

warnings.filterwarnings("ignore")

def run_memory(data: pd.DataFrame,
               fix_start: bool,
               estimation_window: int,
               clustering_method: str,
               k_opt_method: str) -> dict:
    
    clustering = Clustering()
    
    pbar = tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window)
    all_clusters = []
    all_clusters_centers_easy = []
    all_clusters_centers_hard = []
    for step in pbar:

        if fix_start or (step == 0):
            start = 0
        else:
            start += 1

        # subset train data
        train = data.iloc[start:(estimation_window + step), :]

        # compute clusters for easy days
        clusters, clusters_centers = clustering.compute_clusters(data=train,
                                                                 method=clustering_method,
                                                                 k=2,
                                                                 k_opt_method=k_opt_method,
                                                                 similarity_method="euclidean")

        # save easy clusters centers
        if step == 0:
            clusters_centers_df = pd.DataFrame(clusters_centers.T, index=train.index)
            all_clusters_centers_easy.append(clusters_centers_df)
        else:
            clusters_centers_df = pd.DataFrame(clusters_centers.T[-1], columns=[train.index[-1]]).T
            all_clusters_centers_easy.append(clusters_centers_df)

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
        clusters, clusters_centers = clustering.compute_clusters(data=train_hard,
                                                                 method=clustering_method,
                                                                 k_opt_method=k_opt_method,
                                                                 similarity_method="cosine")
        
        # save hard clusters centers
        if step == 0:
            clusters_centers_df = pd.DataFrame(clusters_centers.T, index=train_hard.index)
            all_clusters_centers_hard.append(clusters_centers_df)
        else:
            clusters_centers_df = pd.DataFrame(clusters_centers.T[-1], columns=[train_hard.index[-1]]).T
            all_clusters_centers_hard.append(clusters_centers_df)
        
        train_hard["cluster"] = clusters
        train_easy["cluster"] = clusters.max() + 1

        # merge easy and hard clusters
        train = pd.concat([train_easy, train_hard]).sort_index()

        all_clusters.append(train[["cluster"]].rename(columns={"cluster": f"cluster_step{step}"}))
        pbar.set_description(f"Building memory using window: {step}")

    all_clusters_df = pd.concat(all_clusters, axis=1)
    all_clusters_centers_easy_df = pd.concat(all_clusters_centers_easy, axis=0)
    all_clusters_centers_hard_df = pd.concat(all_clusters_centers_hard, axis=0)

    results = {

        "all_clusters": all_clusters_df,
        "all_clusters_centers_easy": all_clusters_centers_easy_df,
        "all_clusters_centers_hard": all_clusters_centers_hard_df

    }

    return results

        

