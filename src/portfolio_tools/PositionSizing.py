import pandas as pd

class PositionSizing:
    def __init__(self):
        pass

    def positions_from_forecasts(self,
                                 forecasts: pd.Series,
                                 num_assets_to_select: int,
                                 strategy_type: str,
                                 next_regime: int = None):
        """
        Mapping from forecasts to positions to be used in a portfolio.

        Parameters

        forecasts : pd.Series
            Forecasts for each asset.
        num_assets_to_select : int
            Number of assets to select.
        strategy_type : str
            Strategy type. Can be 'lo', 'lns', 'los', or 'm'.
            'lo' is long only, 'lns' is long and short, 'los' is long or short depending on the conviction, and 'm' is mixed.
        next_regime : int
            Next regime to consider. Only used for strategy_type = 'm'.

        Returns

        positions : pd.Series
            Positions for each asset
        """

        forecasts = forecasts.sort_values(ascending=False)

        # adjustment for strategy type = 'm' (mixed)
        if strategy_type == 'm':
            if not next_regime == 0:
                strategy_type = 'lo'
            else:
                strategy_type = 'los'             
        
        if strategy_type == 'lns':
            long_positions = forecasts.sort_values(ascending=False)[:num_assets_to_select]
            short_positions = forecasts.sort_values(ascending=True)[:num_assets_to_select]
            total_positions = pd.concat([long_positions, short_positions]).sort_index().index
            positions = pd.Series([forecasts[i] if i in total_positions else 0.0 for i in forecasts.index], index=forecasts.index)
            positions = positions / positions.abs().sum()
        elif strategy_type == 'lo':
            n_pos = (forecasts > 0).sum()
            cur_num_assets_to_select = min(num_assets_to_select, n_pos)
            if cur_num_assets_to_select == 0:
                selected_assets = forecasts.index[:num_assets_to_select]
                positions = pd.Series([0 for i in range(num_assets_to_select)], index=selected_assets)
            else:
                selected_assets = forecasts.index[:cur_num_assets_to_select]
                positions = pd.Series([(forecasts[i] / forecasts[:cur_num_assets_to_select].sum()) for i in range(cur_num_assets_to_select)], index=selected_assets)
        elif strategy_type == 'los':
            es_abs = forecasts[forecasts.abs().sort_values(ascending=False).index[:num_assets_to_select]]
            positions = pd.Series([(es_abs.abs()[i] / es_abs.abs().sum()) if es_abs[i] >= 0 else -1 * (es_abs.abs()[i] / es_abs.abs().sum()) for i in range(num_assets_to_select)], index=es_abs.index)
        else:
            raise ValueError(f"Strategy type {strategy_type} is not supported. Please choose from 'lo', 'lns', 'los', or 'm'.")

        return positions