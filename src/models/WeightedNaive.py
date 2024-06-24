import pandas as pd
import numpy as np

from utils.activation_functions import sigmoid

class WeightedNaive:
    def __init__(self,
                 num_assets_to_select: int,
                 strategy_type: str = 'mixed',
                 **kwargs):
        self.num_assets_to_select = num_assets_to_select
        self.strategy_type = strategy_type

    def forward(self,
                returns: pd.DataFrame,
                regimes: pd.DataFrame,
                current_regime: int,
                transition_prob: np.ndarray,
                regime_prob: np.ndarray,
                **kwargs):

        TEMPERATURE = 0.85
        regime_prob_exp = np.exp(((regime_prob - regime_prob.mean()) / regime_prob.std()) / TEMPERATURE)
        next_regime_dist = np.matmul(regime_prob_exp / regime_prob_exp.sum(), transition_prob)[0]
        weights_next_regime = next_regime_dist

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

            if self.strategy_type == 'long_only':
                expected_sharpe = expected_sharpe[expected_sharpe > 0]

            expected_sharpe = expected_sharpe.reset_index()
            expected_sharpe.columns = ["asset", "sharpe"]
            expected_sharpe["prob"] = next_regime_weight
            expected_sharpe["regime"] = next_regime
            expected_sharpes.append(expected_sharpe)
        expected_sharpes = pd.concat(expected_sharpes)
        expected_sharpes["weighted_sharpe"] = expected_sharpes["sharpe"] * expected_sharpes["prob"]

        if self.strategy_type == 'long_only':
            expected_sharpes = expected_sharpes.groupby("asset").sum("weighted_sharpe").sort_values(by="weighted_sharpe", ascending=False)
            positions = expected_sharpes["weighted_sharpe"] / expected_sharpes["weighted_sharpe"].sum()
        else:
            expected_sharpe_longs = expected_sharpes[expected_sharpes["sharpe"] > 0]
            expected_sharpe_shorts = expected_sharpes[expected_sharpes["sharpe"] < 0]

            expected_sharpe_longs = expected_sharpe_longs.groupby("asset").sum("weighted_sharpe").sort_values(by="weighted_sharpe", ascending=False)
            positions_longs = expected_sharpe_longs["weighted_sharpe"] / expected_sharpe_longs["weighted_sharpe"].sum()

            expected_sharpe_shorts = expected_sharpe_shorts.groupby("asset").sum("weighted_sharpe").sort_values(by="weighted_sharpe", ascending=False)
            positions_shorts = (expected_sharpe_shorts["weighted_sharpe"] / expected_sharpe_shorts["weighted_sharpe"].sum() * -1)

            positions = pd.concat([positions_longs, positions_shorts])
            positions = positions.reset_index().groupby("asset").sum().sort_values(by="weighted_sharpe")
            positions_longs = positions[positions["weighted_sharpe"] > 0]
            positions_longs = positions_longs / positions_longs.sum()
            positions_shorts = positions[positions["weighted_sharpe"] < 0]
            positions_shorts = (positions_shorts / positions_shorts.sum() * -1)
            positions = pd.concat([positions_longs, positions_shorts])["weighted_sharpe"]

        return positions