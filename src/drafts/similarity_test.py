
import pandas as pd
import torch

WINDOW_SIZE = 3
STEP_SIZE = 1
TOP_K = 10

if __name__ == "__main__":
    # get data and turn it into tensor data
    fredmd_data = pd.read_csv('src/data/inputs/fredmd_transf.csv')
    fred_dates = fredmd_data['date'].to_numpy()
    del fredmd_data['date']
    fred_tensor = torch.tensor(fredmd_data.to_numpy(), dtype=torch.float)

    # get latest window
    latest_window = fred_tensor[-1 * WINDOW_SIZE:, :]

    # loop through and get similarities
    scores = []
    for step in range(0, fred_tensor.shape[0] - WINDOW_SIZE + 1, STEP_SIZE):
        start_ind = step
        end_ind = start_ind + WINDOW_SIZE
        cur_window = fred_tensor[start_ind:end_ind, :]
        cur_similarity = torch.norm((latest_window - cur_window).nan_to_num())
        scores.append([start_ind, fred_dates[start_ind], float(cur_similarity)])

    # clean up top k scores
    scores = sorted(scores, key=lambda x: x[2])
    scores = scores[0:TOP_K]
    print(scores)
