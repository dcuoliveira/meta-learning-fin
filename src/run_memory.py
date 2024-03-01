import argparse
import pandas as pd
import os

from learning.memory import run_memory
from utils.conn_data import save_pickle

parser = argparse.ArgumentParser(description="Run forecast.")

parser.add_argument("--estimation_window", type=int, default=12 * 4)
parser.add_argument("--fix_start", type=bool, default=True)
parser.add_argument("--clustering_method", type=str, default="kmeans")
parser.add_argument("--k_opt_method", type=str, default=None)
parser.add_argument("--memory_input", type=str, default="fredmd_transf")
parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))
parser.add_argument("--outputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "outputs"))

if __name__ == "__main__":

    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.inputs_path, f'{args.memory_input}.csv'))
    
    # fix dates
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    # compute moving average
    data = data.rolling(window=12).mean()

    # drop missing values
    data = data.dropna()

    memory = run_memory(data=data,
                        fix_start=args.fix_start,
                        estimation_window=args.estimation_window,
                        k_opt_method=args.k_opt_method,
                        clustering_method=args.clustering_method)
    
    results = {

        "memory": memory,
        "estimation_window": args.estimation_window,
        "fix_start": args.fix_start,
        "k_opt_method": args.k_opt_method,
        "clustering_method": args.clustering_method

    }

    # check if results folder exists
    if not os.path.exists(os.path.join(args.outputs_path, args.clustering_method)):
        os.makedirs(os.path.join(args.outputs_path, args.clustering_method))
    
    # save results
    save_path = os.path.join(args.outputs_path,
                             args.clustering_method,
                             f"results_{args.k_opt_method}.pkl")
    save_pickle(path=save_path, obj=results)