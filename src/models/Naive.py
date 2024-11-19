import pandas as pd
import numpy as np

from portfolio_tools.PositionSizing import PositionSizing

class Naive(PositionSizing):
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
                random_regime: bool = False,
                **kwargs):

        TEMPERATURE = 0.85
        regime_prob_exp = np.exp(((regime_prob - regime_prob.mean()) / regime_prob.std()) / TEMPERATURE)
        next_regime_dist = np.matmul(regime_prob_exp / regime_prob_exp.sum(), transition_prob)[0]
        next_regime = np.argmax(next_regime_dist)
        if random_regime:
            next_regime = np.random.choice(list(range(6)))

        labelled_returns = pd.merge(returns, regimes, left_index=True, right_index=True)

        cluster_name = labelled_returns.columns[-1]

        # select dates that match the next regime
        next_regime_returns = labelled_returns[labelled_returns[cluster_name] == next_regime].drop(cluster_name, axis=1)

        # compute expected sharpe ratio on the next regime
        forecasts = next_regime_returns.mean() / next_regime_returns.std()

        # generate positions
        positions = self.positions_from_forecasts(forecasts=forecasts,
                                                  num_assets_to_select=self.num_assets_to_select,
                                                  strategy_type=self.strategy_type,
                                                  next_regime=next_regime,)

        return positions
