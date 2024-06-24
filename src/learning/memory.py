
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import itertools

from models.Clustering import Clustering

warnings.filterwarnings("ignore")

BEST_K = 5

def run_memory(data: pd.DataFrame,
               fix_start: bool,
               estimation_window: int,
               clustering_method: str,
               k_opt_method: str,) -> dict:

    low_pass_clustering = Clustering(similarity_method='euclidean')
    clustering = Clustering(similarity_method='cosine')
    
    # we have already found that 5 is the optimal value for most recent data
    # k = clustering.compute_k_opt(data=data, k_opt_method=k_opt_method)
    k = BEST_K
    permutations = list(itertools.permutations(range(k)))
    
    pbar = tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window)
    all_clusters = []
    out_centroids = []
    all_probs = []
    prev_centroid_map = None
    for step in pbar:
        if fix_start:
            start = 0
        else:
            start = step

        # subset train data
        train = data.iloc[start:step + estimation_window, :]

        # compute clusters for easy days
        low_pass_k = 2
        cutoff_date = pd.Timestamp('2021-06-01')
        #cutoff_date = pd.Timestamp('2021-11-01')
        if train.index.to_list()[-1] > cutoff_date:
            low_pass_k = 3
        clusters, euc_clusters, euc_probs = low_pass_clustering.compute_clusters(data=train, extra_data=train, method=clustering_method, k=low_pass_k, k_opt_method=k_opt_method)

        # find the cluster with the most observations and set as "hard days"
        train["cluster"] = clusters
        cluster_sizes = []
        for i in range(low_pass_k):
            temp = train.loc[:, "cluster"] == i
            temp = temp[temp == True]
            cluster_sizes.append(len(temp))
        largest_cluster = np.argmax(cluster_sizes)

        ## subset train data
        train_hard = train[train.loc[:, "cluster"] == largest_cluster]
        train_easy = train[train.loc[:, "cluster"] != largest_cluster]
        temp = []
        for i in range(low_pass_k):
            if i == largest_cluster:
                temp.insert(0, euc_clusters[i])
            else:
                temp.append(euc_clusters[i])
        euc_clusters = np.array(temp)
        euc_probs = np.concatenate([euc_probs[:, largest_cluster].reshape(-1, 1), np.delete(euc_probs, largest_cluster, axis=1)], axis=1)

        # compute clusters for hard days
        clusters, hard_centroids, hard_probs = clustering.compute_clusters(data=train_hard, extra_data=train, method=clustering_method, k=k)

        train_easy["cluster"] = 0
        train_hard["cluster"] = clusters + 1
        # merge easy and hard clusters
        train = pd.concat([train_easy, train_hard]).sort_index()

        # find the best labeling permutation
        if prev_centroid_map is None:
            prev_centroid_map = train["cluster"].values[:-1]
        cur_min = np.inf
        best_perm = None
        for perm in permutations:
            cur_perm = np.array(perm) + 1
            cur_perm = np.insert(cur_perm, 0, 0)
            cur_count = (train["cluster"].replace(range(0, k + 1), cur_perm)[:-1] != prev_centroid_map).sum()
            cur_count = cur_count / (len(train) - 1)
            if cur_count < cur_min:
                cur_min = cur_count
                best_perm = cur_perm
        train["cluster"] = train["cluster"].replace(range(0, k + 1), best_perm)
        train["cluster"] = train["cluster"].astype(float)
        perm_indices = np.array(range(0, k + 1))
        for i in range(0, k + 1):
            perm_indices[i] = np.where(best_perm == i)[0]
        hard_centroids = hard_centroids[perm_indices[1:] - 1]
        hard_probs = hard_probs[:, perm_indices[1:] - 1]
        prev_centroid_map = train["cluster"].values

        # save clusters
        all_clusters.append(train[["cluster"]].rename(columns={"cluster": f"cluster_step{step}"}))
        out_centroids.append([euc_clusters, hard_centroids])

        # calculate final probabilities
        a = hard_probs.max(axis=1).reshape(-1, 1) / np.log(0.5)
        final_probs = euc_probs[:, 0].reshape(-1, 1) - euc_probs[:, 1:].reshape(euc_probs.shape[0], -1).max(axis=1).reshape(-1, 1)
        final_probs = (1 - final_probs) / 2
        final_probs = a * (np.log(1 - final_probs))
        final_probs = np.concatenate([final_probs, hard_probs], axis=1)
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
        all_probs.append(final_probs)
        pbar.set_description(f"Building memory using window: {step}")

    all_clusters_df = pd.concat(all_clusters, axis=1)
    
    probs_out = {}
    for i in range(all_clusters_df.columns.size):
        probs_out[all_clusters_df[f"cluster_step{i}"].dropna().index[-1].strftime("%Y-%m-%d")] = all_probs[i][-1].reshape((1, -1))

    return all_clusters_df, out_centroids, probs_out

def compute_transition_matrix(data: pd.DataFrame):
    output = {}
    for i in range(data.columns.size):
        cur_data = data[f"cluster_step{i}"]
        cur_date = cur_data.dropna().index[-1].strftime("%Y-%m-%d")
        transition_matrix = np.zeros((6, 6))
        for j in range(len(cur_data) - 1):
            cur = cur_data.iloc[j]
            nex = cur_data.iloc[j + 1]
            if np.isnan(nex):
                break
            transition_matrix[int(cur), int(nex)] += 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        output[cur_date] = transition_matrix.copy()
        
    return output
