import pandas as pd
import numpy as np

class Naive:
    def __init__(self, num_assets_to_select: int, long_only: bool = False):
        self.num_assets_to_select = num_assets_to_select
        self.long_only = long_only

    def forward(self, returns: pd.DataFrame, regimes: pd.DataFrame, current_regime: int, next_regime: int):

        labelled_returns = pd.merge(returns, regimes, left_index=True, right_index=True)
        
        cluster_name = labelled_returns.columns[-1]

        # select dates that match the next regime
        next_regime_returns = labelled_returns[labelled_returns[cluster_name] == next_regime].drop(cluster_name, axis=1)

        # compute expected sharpe ratio on the next regime
        expected_sharpe = next_regime_returns.mean() / next_regime_returns.std()
        expected_sharpe = expected_sharpe.sort_values(ascending=False)

        # select top/bottom assets
        if self.long_only:
            selected_assets = expected_sharpe.index[:self.num_assets_to_select]
            positions = pd.Series([self.num_assets_to_select * (expected_sharpe[i] / expected_sharpe[:self.num_assets_to_select].sum()) for i in range(self.num_assets_to_select)], index=selected_assets)
        else:
            selected_assets = expected_sharpe.index[:self.num_assets_to_select].append(expected_sharpe.index[-self.num_assets_to_select:])
            es_abs = expected_sharpe[expected_sharpe.abs().sort_values(ascending=False).index[:self.num_assets_to_select]]
            positions = pd.Series([self.num_assets_to_select * (es_abs.abs()[i] / es_abs.abs().sum()) if es_abs[i] >= 0 else -1 * self.num_assets_to_select * (es_abs.abs()[i] / es_abs.abs().sum()) for i in range(self.num_assets_to_select)], index=es_abs.index)
        return positions