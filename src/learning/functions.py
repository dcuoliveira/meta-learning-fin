
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import itertools
from models.Clustering import Clustering

warnings.filterwarnings("ignore")

def run_memory(data: pd.DataFrame,
               fix_start: bool,
               estimation_window: int,
               clustering_method: str,
               k_opt_method: str) -> dict:

    low_pass_clustering = Clustering(similarity_method='euclidean-distance')
    clustering = Clustering(similarity_method=similarity_method)
    #k = clustering.compute_k_opt(data=data, k_opt_method=k_opt_method)
    k = 5
    permutations = list(itertools.permutations(range(k)))
    
    pbar = tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window)
    all_clusters = []
    prev_centroid_map = None
    for step in pbar:
        pbar.set_description(f"Building memory using window: {step}")
        if fix_start:
            start = 0
        else:
            start = step

        # subset train data
        train = data.iloc[start:step + estimation_window, :]

        # compute clusters for easy days
        low_pass_k = 2
        if pd.Timestamp('2021-06-01') < train.index.to_list()[-1]:
            low_pass_k = 3
        clusters, _ = low_pass_clustering.compute_clusters(data=train, method=clustering_method, k=low_pass_k, k_opt_method=k_opt_method)

        # find the cluster with the most observations and set as "hard days"
        train["cluster"] = clusters

        lengths = []
        for i in range(low_pass_k):
            temp = train.loc[:, "cluster"] == i
            temp = temp[temp == True]
            lengths.append(len(temp))
        argmax = np.argmax(lengths)

        ## subset train data
        train_hard = train[train.loc[:, "cluster"] == argmax]
        train_easy = train[train.loc[:, "cluster"] != argmax]

        # compute clusters for hard days
        clusters, _ = clustering.compute_clusters(data=train_hard, method=clustering_method, k=k)
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
        prev_centroid_map = train["cluster"].values

        # save clusters
        all_clusters.append(train[["cluster"]].rename(columns={"cluster": f"cluster_step{step}"}))
    all_clusters_df = pd.concat(all_clusters, axis=1)
    return all_clusters_df

def compute_transition_matrix(data: pd.DataFrame):
    output = []
    for i in range(data.columns.size):
        cur_data = data[f"cluster_step{i}"]
        transition_matrix = np.zeros((6, 6))
        for j in range(len(cur_data) - 1):
            cur = cur_data.iloc[j]
            nex = cur_data.iloc[j + 1]
            if np.isnan(nex):
                break
            transition_matrix[int(cur), int(nex)] += 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        output.append(transition_matrix.copy())
    return output