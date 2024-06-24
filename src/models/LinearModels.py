import pandas as pd
import numpy as np

from models.ModelWrappers import LinearRegressionWrapper, LassoWrapper, RidgeWrapper
from learning.hyper_params_search import hyper_params_search

class LinearModels:
    def __init__(self,
                 strategy_type: str, 
                 portfolio_method: str,
                 num_assets_to_select: int,
                 cv_search_type: str,
                 cv_split_type: str,
                 cv_folds: int,
                 cv_iters: int,
                 **kwargs):
        
        self.strategy_type = strategy_type
        self.num_assets_to_select = num_assets_to_select
        self.cv_search_type = cv_search_type
        self.cv_split_type = cv_split_type
        self.cv_folds = cv_folds
        self.cv_iters = cv_iters
        self.seed = 2294

        if portfolio_method == 'linear-ols':
            self.model = LinearRegressionWrapper()
        elif portfolio_method == 'linear-lasso':
            self.model = LassoWrapper()
        elif portfolio_method == 'linear-ridge':
            self.model = RidgeWrapper()
        else:
            raise ValueError(f"Model type {portfolio_method} not recognized.")

    def forward(self,
                returns: pd.DataFrame,
                features: pd.DataFrame,
                test_features: pd.DataFrame,
                regimes: pd.DataFrame,
                current_regime: int,
                transition_prob: np.ndarray,
                regime_prob: np.ndarray,):

        TEMPERATURE = 0.85
        regime_prob_exp = np.exp(((regime_prob - regime_prob.mean()) / regime_prob.std()) / TEMPERATURE)
        next_regime_dist = np.matmul(regime_prob_exp / regime_prob_exp.sum(), transition_prob)[0]
        next_regimes = np.argsort(next_regime_dist)[::-1]

        labelled_returns = pd.merge(returns, regimes, left_index=True, right_index=True)
        
        cluster_name = labelled_returns.columns[-1]

        all_predictions = []
        for next_regime in next_regimes:

            # select dates that match the next regime
            next_regime_returns = labelled_returns[labelled_returns[cluster_name] == next_regime].drop(cluster_name, axis=1)

            for target in next_regime_returns.columns:

                # training data
                train_df = pd.merge(next_regime_returns[[target]], features, left_index=True, right_index=True)

                if train_df.shape[0] < self.cv_folds + 1:
                    continue

                # search for the best hyperparameters
                model_search = hyper_params_search(df=train_df,
                                                wrapper=self.model,
                                                search_type=self.cv_search_type,
                                                n_iter=self.cv_iters,
                                                n_splits=self.cv_folds,
                                                n_jobs=-1,
                                                seed=self.seed,
                                                target_name=target)

                X_test = test_features.values
                test_prediction = model_search.best_estimator_.predict(X_test)
                result = pd.DataFrame({"etf": target,
                                       "regime": next_regime,
                                       "weight": next_regime_dist[next_regime],
                                       "prediction": test_prediction})
                all_predictions.append(result)
                
        if len(all_predictions) == 0:
            return pd.Series(0, index=returns.columns)
        else:
            all_predictions_df = pd.concat(all_predictions, axis=0)
            all_predictions_df["weight"] = all_predictions_df["weight"] / all_predictions_df["weight"].iloc[::all_predictions_df["etf"].nunique()].sum()
            all_predictions_df["weighted_prediction"] = all_predictions_df["weight"] * all_predictions_df["prediction"]
            expected_sharpe = all_predictions_df.groupby(["etf"]).sum()[["weighted_prediction"]]["weighted_prediction"]
            expected_sharpe = expected_sharpe.sort_values(ascending=False)

            if self.strategy_type == 'long_only':
                do_long_only = True
            elif self.strategy_type == 'long_short':
                do_long_only = False
            elif self.strategy_type == 'mixed':
                do_long_only = not (np.argmax(next_regime_dist) == 0)

            # select top/bottom assets
            if do_long_only:
                n_pos = (expected_sharpe > 0).sum()
                cur_num_assets_to_select = min(self.num_assets_to_select, n_pos)
                selected_assets = expected_sharpe.index[:cur_num_assets_to_select]
                positions = pd.Series([(expected_sharpe[i] / expected_sharpe[:cur_num_assets_to_select].sum()) for i in range(cur_num_assets_to_select)], index=selected_assets)
            else:
                es_abs = expected_sharpe[expected_sharpe.abs().sort_values(ascending=False).index[:self.num_assets_to_select]]
                positions = pd.Series([(es_abs.abs()[i] / es_abs.abs().sum()) if es_abs[i] >= 0 else -1 * (es_abs.abs()[i] / es_abs.abs().sum()) for i in range(self.num_assets_to_select)], index=es_abs.index)

        return positions