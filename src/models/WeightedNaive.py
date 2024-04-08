import pandas as pd
import numpy as np

from utils.activation_functions import sigmoid

class WeightedNaive:
    def __init__(self, num_assets_to_select: int, long_only: bool = False):
        self.num_assets_to_select = num_assets_to_select
        self.long_only = long_only

    def forward(self, returns: pd.DataFrame, regimes: pd.DataFrame, current_regime: int, transition_prob: np.ndarray):

        weights_next_regime = transition_prob[current_regime, :]

        labelled_returns = pd.merge(returns, regimes, left_index=True, right_index=True)
        cluster_name = labelled_returns.columns[-1]

        expected_sharpes = []
        for next_regime in range(len(weights_next_regime)):
            next_regime_weight = weights_next_regime[next_regime]

            if next_regime_weight == 0:
                continue

            # select dates that match the next regime
            next_regime_returns = labelled_returns[labelled_returns[cluster_name] == next_regime].drop(cluster_name, axis=1)

            # compute expected sharpe ratio on the next regime
            expected_sharpe = next_regime_returns.mean() / next_regime_returns.std()
            expected_sharpe = expected_sharpe.sort_values(ascending=False)

            if self.long_only:
                expected_sharpe = expected_sharpe[expected_sharpe > 0]

            expected_sharpe = expected_sharpe.reset_index()
            expected_sharpe.columns = ["asset", "sharpe"]
            expected_sharpe["prob"] = next_regime_weight
            expected_sharpe["regime"] = next_regime
            expected_sharpes.append(expected_sharpe)
        expected_sharpes = pd.concat(expected_sharpes)
        expected_sharpes["weighted_sharpe"] = expected_sharpes["sharpe"] * expected_sharpes["prob"]

        if self.long_only:
            expected_sharpes = expected_sharpes.groupby("asset").sum("weighted_sharpe").sort_values(by="weighted_sharpe", ascending=False)
            positions = expected_sharpes["weighted_sharpe"] / expected_sharpes["weighted_sharpe"].sum()
        else:
            raise NotImplementedError

        return positions