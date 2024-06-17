import pandas as pd
import numpy as np

class Naive:
    def __init__(self, num_assets_to_select: int, strategy_type: str = 'mixed', **kwargs):
        self.num_assets_to_select = num_assets_to_select
        self.strategy_type = strategy_type

    def forward(self, returns: pd.DataFrame, regimes: pd.DataFrame, current_regime: int, transition_prob: np.ndarray, **kwargs):

        next_regime = np.argmax(transition_prob[current_regime, :])

        labelled_returns = pd.merge(returns, regimes, left_index=True, right_index=True)
        
        cluster_name = labelled_returns.columns[-1]

        # select dates that match the next regime
        next_regime_returns = labelled_returns[labelled_returns[cluster_name] == next_regime].drop(cluster_name, axis=1)

        # compute expected sharpe ratio on the next regime
        expected_sharpe = next_regime_returns.mean() / next_regime_returns.std()
        expected_sharpe = expected_sharpe.sort_values(ascending=False)

        if self.strategy_type == 'long_only':
            do_long_only = True
        elif self.strategy_type == 'long_short':
            do_long_only = False
        elif self.strategy_type == 'mixed':
            do_long_only = not (next_regime == 0)
            
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