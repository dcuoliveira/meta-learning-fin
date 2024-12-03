import argparse
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

from learning.memory import run_memory, compute_transition_matrix
from learning.forecasts import run_forecasts
from utils.conn_data import save_pickle, load_pickle
from utils.parsers import str_2_bool
from models.ModelUtils import ModelUtils as mu

parser = argparse.ArgumentParser(description="Run forecast.")

parser.add_argument("--estimation_window", type=int, default=12 * 4)
parser.add_argument("--fix_start", type=str, default="True")
parser.add_argument("--clustering_method", type=str, default="kmeans")
parser.add_argument("--k_opt_method", type=str, default="elbow")
parser.add_argument("--memory_input", type=str, default="fredmd_transf")
parser.add_argument("--forecast_input", type=str, default="wrds_etf_returns")
parser.add_argument("--portfolio_method", type=str, default="linear-ridge", choices=["bl", "mvo", "naive", "weighted-naive", "linear-ols", "linear-ridge", "linear-lasso"])
parser.add_argument("--cv_split_type", type=str, default="tscv", choices=["tscv", "cv"])
parser.add_argument("--cv_search_type", type=str, default="random", choices=["random", "grid"])
parser.add_argument("--cv_folds", type=int, default=5)
parser.add_argument("--cv_iters", type=int, default=20)
parser.add_argument("--strategy_type", type=str, default="lo", choices=["lo", "lns", "los", "m"])
parser.add_argument("--num_assets_to_select", type=int, default=4)
parser.add_argument("--random_regime", type=str, default="False")
parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))
parser.add_argument("--outputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "outputs"))


def run_new_memory(inputs_path, estimation_window, k_opt_method, clustering_method, fix_start):
    data_factors = pd.read_csv(os.path.join(inputs_path, "fredmd_factors_raw.csv"))
    transformation_codes = data_factors.iloc[0]
    data_factors = data_factors.drop(0)
    transformation_codes = transformation_codes.to_dict()
    del transformation_codes['sasdate']

    small = 1e-6
    for column in data_factors.columns:
        if column in transformation_codes:
            match int(transformation_codes[column]):
                case 1:
                    data_factors[column] = data_factors[column]

                case 2: # First difference: x(t)-x(t-1)
                    data_factors[column] = data_factors[column].diff()

                case 3: # Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
                    data_factors[column] = data_factors[column].diff().diff()

                case 4: # Natural log: ln(x)
                    data_factors[column] = data_factors[column].apply(lambda x: np.log(x) if x > small else None)

                case 5: # First difference of natural log: ln(x)-ln(x-1)
                    data_factors[column] = data_factors[column].apply(lambda x: np.log(x) if x > small else None)
                    data_factors[column] = data_factors[column].diff()

                case 6: # Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
                    data_factors[column] = data_factors[column].apply(lambda x: np.log(x) if x > small else None)
                    data_factors[column] = data_factors[column].diff().diff()

                case 7: # First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
                    data_factors[column] = data_factors[column].pct_change()
                    data_factors[column] = data_factors[column].diff()

    data_factors = data_factors.drop([1, 2]).reset_index(drop=True)

    data_factors = data_factors.ffill()
    data_factors = data_factors.fillna(0.0)

    data_factors['sasdate'] = pd.to_datetime(data_factors['sasdate'], format='%m/%d/%Y')
    data_factors = data_factors.rename(columns={'sasdate': 'date'})
    data_factors = data_factors.set_index('date')

    #data_factors = data_factors.rolling(window=12).mean()
    #data_factors = data_factors.dropna()

    df_normalized = data_factors

    # You must normalize the data before applying the fit method
    df_normalized=(df_normalized - df_normalized.mean()) / df_normalized.std()
    pca = PCA(n_components=df_normalized.shape[1])
    pca.fit(df_normalized)

    # Reformat and view results
    loadings = pd.DataFrame(pca.components_.T,
    columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
    index=df_normalized.columns)
    # print(loadings)

    DESIRE_EXPLAINED_VARIANCE = 0.95
    total_explained_variance = 0.0
    for i, x in enumerate(pca.explained_variance_ratio_):
        total_explained_variance += x
        if total_explained_variance >= DESIRE_EXPLAINED_VARIANCE:
            # print(f"Number of components to explain {DESIRE_EXPLAINED_VARIANCE * 100}% variance: {i+1}")
            break
    n_components = i+1
    # %config InlineBackend.figure_format = 'retina'
    # plot.plot([DESIRE_EXPLAINED_VARIANCE] * len(pca.explained_variance_ratio_), 'r--')
    # plot.plot(pca.explained_variance_ratio_.cumsum())
    # plot.title('PCA Explained Variance')
    # plot.ylabel('Explained Variance')
    # plot.xlabel('Total # of Components')
    # plot.legend(['95% Variance Explained', 'Cumulative Explained Variance'])
    # plot.show()

    # Use the top n components to transform the data
    pca = PCA(n_components=df_normalized.shape[1])
    pca.fit(df_normalized)
    df_transformed = pd.DataFrame(pca.transform(df_normalized),
    columns=['PC%s' % _ for _ in range(df_normalized.shape[1])],
    index=df_normalized.index)
    df_transformed = df_transformed[['PC%s' % _ for _ in range(n_components)]]

    data = df_transformed

    memory, centroids, regimes_probs = run_memory(
        data=data,
        fix_start=fix_start,
        estimation_window=estimation_window,
        k_opt_method=k_opt_method,
        clustering_method=clustering_method,
    )

    results = {
        "memory": memory,
        "estimation_window": estimation_window,
        "fix_start": fix_start,
        "k_opt_method": k_opt_method,
        "clustering_method": clustering_method
    }

    return results, centroids, regimes_probs, df_normalized


if __name__ == "__main__":

    args = parser.parse_args()
    args.fix_start = str_2_bool(args.fix_start)
    args.random_regime = str_2_bool(args.random_regime)

    if args.strategy_type == "lo":
        long_only_tag = "lo"
    elif args.strategy_type == "lns":
        long_only_tag = "lns"
    elif args.strategy_type == "los":
        long_only_tag = "los"
    elif args.strategy_type == "m":
        long_only_tag = "mx"
    
    memory_results, centroids, regimes_probs, memory_data = run_new_memory(inputs_path=args.inputs_path,
                                                                           estimation_window=args.estimation_window,
                                                                           k_opt_method=args.k_opt_method,
                                                                           clustering_method=args.clustering_method,
                                                                           fix_start=args.fix_start)
    regimes = memory_results["memory"]

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
                              random_regime=args.random_regime,)

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

    file_name = f"results_{long_only_tag}"
    if args.num_assets_to_select is not None:
        file_name += f"_{args.num_assets_to_select}"
    if args.random_regime:
        file_name += "_rand"
    file_name += ".pkl"

    # save results
    save_path = os.path.join(args.outputs_path,
                             args.portfolio_method,
                             file_name)
    save_pickle(path=save_path, obj=results)
