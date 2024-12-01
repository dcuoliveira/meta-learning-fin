import pandas as pd
import numpy as np

from models.ModelWrappers import LinearRegressionWrapper, LassoWrapper, RidgeWrapper
from learning.hyper_params_search import hyper_params_search
from portfolio_tools.PositionSizing import PositionSizing

class LinearModels(PositionSizing):
    def __init__(self,
                 num_assets_to_select: int,
                 strategy_type: str, 
                 portfolio_method: str,
                 cv_search_type: str,
                 cv_split_type: str,
                 cv_folds: int,
                 cv_iters: int,
                 **kwargs):

        self.num_assets_to_select = num_assets_to_select
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
                regime_prob: np.ndarray,
                random_regime: bool = False,):

        TEMPERATURE = 0.85
        regime_prob_exp = np.exp(((regime_prob - regime_prob.mean()) / regime_prob.std()) / TEMPERATURE)
        next_regime_dist = np.matmul(regime_prob_exp / regime_prob_exp.sum(), transition_prob)[0]
        if random_regime:
            next_regime_dist = np.ones((next_regime_dist.shape[0], )) * 1/next_regime_dist.shape[0]
        next_regimes = np.argsort(next_regime_dist)[::-1]

        labelled_returns = pd.merge(returns, regimes, left_index=True, right_index=True)

        cluster_name = labelled_returns.columns[-1]

        forecasts = []
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
                                                target_name=target,)

                X_test = test_features.values
                test_prediction = model_search.best_estimator_.predict(X_test)
                result = pd.DataFrame({"etf": target,
                                       "regime": next_regime,
                                       "weight": next_regime_dist[next_regime],
                                       "prediction": test_prediction})
                forecasts.append(result)

        if len(forecasts) == 0:
            return pd.Series(0, index=returns.columns)

        forecasts = pd.concat(forecasts, axis=0)

        forecasts["weight"] = forecasts["weight"] / forecasts["weight"].iloc[::forecasts["etf"].nunique()].sum()
        forecasts["weighted_prediction"] = forecasts["weight"] * forecasts["prediction"]
        forecasts = forecasts.groupby(["etf"]).sum()[["weighted_prediction"]]["weighted_prediction"]

        # generate positions
        positions = self.positions_from_forecasts(forecasts=forecasts,
                                                  num_assets_to_select=self.num_assets_to_select,
                                                  strategy_type=self.strategy_type,
                                                  next_regime=next_regimes[0],)

        return positions
