import argparse
import pandas as pd
import os
import numpy as np

from learning.memory import run_memory, compute_transition_matrix
from learning.forecasts import run_forecasts
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool
from models.ModelUtils import ModelUtils as mu

parser = argparse.ArgumentParser(description="Run forecast.")

parser.add_argument("--estimation_window", type=int, default=12 * 4)
parser.add_argument("--fix_start", type=str, default="True")
parser.add_argument("--clustering_method", type=str, default="kmeans")
parser.add_argument("--k_opt_method", type=str, default="elbow")
parser.add_argument("--memory_input", type=str, default="fredmd_transf")
parser.add_argument("--forecast_input", type=str, default="wrds_etf_returns")
parser.add_argument("--portfolio_method", type=str, default="linear-ridge", choices=["naive", "weighted-naive", "linear-ols", "linear-ridge", "linear-lasso"])
parser.add_argument("--cv_split_type", type=str, default="tscv", choices=["tscv", "cv"])
parser.add_argument("--cv_search_type", type=str, default="random", choices=["random", "grid"])
parser.add_argument("--cv_folds", type=int, default=5)
parser.add_argument("--cv_iters", type=int, default=20)
parser.add_argument("--strategy_type", type=str, default="long_only", choices=["long_only", "long_short", "mixed"])
parser.add_argument("--num_assets_to_select", type=int, default=3)
parser.add_argument("--random_regime", type=str, default="False")
parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))
parser.add_argument("--outputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "outputs"))

if __name__ == "__main__":

    args = parser.parse_args()
    args.fix_start = str_2_bool(args.fix_start)
    args.random_regime = str_2_bool(args.random_regime)

    if args.strategy_type == "long_only":
        long_only_tag = "lo"
    elif args.strategy_type == "long_short":
        long_only_tag = "ls"
    elif args.strategy_type == "mixed":
        long_only_tag = "mx"

    # load memory data and preprocess
    memory_data = pd.read_csv(os.path.join(args.inputs_path, f'{args.memory_input}.csv'))
    
    ## fix dates
    memory_data["date"] = pd.to_datetime(memory_data["date"])
    memory_data = memory_data.set_index("date")
    memory_data = memory_data.astype(float)
    
    # fill missing values
    memory_data = memory_data.interpolate(method='linear', limit_direction='forward', axis=0)
    memory_data = memory_data.fillna(method='ffill')
    memory_data = memory_data.fillna(method='bfill')

    ## compute moving average
    memory_data = memory_data.rolling(window=12).mean()

    ## drop missing values
    memory_data = memory_data.dropna()

    # load forecast data and preprocess
    returns = pd.read_csv(os.path.join(args.inputs_path, f'{args.forecast_input}.csv'))
    returns = returns[[col for col in returns.columns if "t+1" not in col]]

    ## fix dates
    returns["date"] = pd.to_datetime(returns["date"])
    returns["date"] = returns["date"] + pd.DateOffset(months=1)
    returns = returns.set_index("date")
    memory_data = memory_data.astype(float)

    ## resample and match memory data dates
    returns = returns.resample("MS").last().ffill()
    returns = pd.merge(returns, memory_data[[memory_data.columns[0]]], left_index=True, right_index=True).drop(memory_data.columns[0], axis=1)

    ## drop missing values
    returns = returns.dropna()

    # build memory
    regimes, centroids, regimes_probs = run_memory(data=memory_data,
                                                   fix_start=args.fix_start,
                                                   estimation_window=args.estimation_window,
                                                   k_opt_method=args.k_opt_method,
                                                   clustering_method=args.clustering_method)
    
    # compute transition probabilities
    regimes_transition_probs = compute_transition_matrix(data=regimes)

    # parse portfolio method
    model = mu.parse_portfolio_method(portfolio_method=args.portfolio_method)
    
    # generate forecasts given memory
    forecasts = run_forecasts(returns=returns,
                              features=memory_data,
                              regimes=regimes,
                              regimes_probs=regimes_probs,
                              transition_probs=regimes_transition_probs,
                              estimation_window=args.estimation_window,
                              model=model,
                              portfolio_method=args.portfolio_method,
                              cv_split_type=args.cv_split_type,
                              cv_search_type=args.cv_search_type,
                              cv_folds=args.cv_folds,
                              cv_iters=args.cv_iters,
                              num_assets_to_select=args.num_assets_to_select,
                              fix_start=args.fix_start,
                              strategy_type=args.strategy_type,
                              random_regime=args.random_regime)

    results = {
        "regimes": regimes,
        "centroids": centroids,
        "regimes_probs": regimes_probs,
        "transition_probs": regimes_transition_probs,
        "model": model,
        "forecasts": forecasts,
        "args": args
    }

    # check if results folder exists
    if not os.path.exists(os.path.join(args.outputs_path, args.portfolio_method)):
        os.makedirs(os.path.join(args.outputs_path, args.portfolio_method))
    
    # save results
    file_name = f"results_{long_only_tag}"
    if args.num_assets_to_select is not None:
        file_name += f"_{args.num_assets_to_select}"
    if args.random_regime:
        file_name += "_rand"
    file_name += ".pkl"
    save_path = os.path.join(args.outputs_path,
                             args.portfolio_method,
                             file_name)
    save_pickle(path=save_path, obj=results)