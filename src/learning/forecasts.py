from tqdm import tqdm
import numpy as np
import pandas as pd

def run_forecasts(returns: pd.DataFrame,
                  regimes: pd.DataFrame,
                  transition_probs: dict,
                  estimation_window: int,
                  model: object,
                  num_assets_to_select: int,
                  fix_start: bool = False,
                  long_only: bool = False):
    
    model_init = model(num_assets_to_select=num_assets_to_select, long_only=long_only)

    pbar = tqdm(range(0, len(returns) - estimation_window, 1), total=len(returns) - estimation_window, desc="Running Forecasts")
    all_positions = []
    for step in pbar:

        if fix_start:
            start = 0
        else:
            start = step

        # subset returns
        current_returns = returns.iloc[start:step + estimation_window, :]
        cur_date = current_returns.index[-1].strftime('%Y-%m-%d')
        pred_date = returns.index[step + estimation_window].strftime('%Y-%m-%d')

        # current Regime Identification
        # find date of last nan value for regimes datafrmae
        regimes_index = regimes.apply(lambda x: x.dropna().index[-1])
        regime_label = regimes_index[regimes_index == cur_date].index[0]
        current_regime_column = regimes.loc[:, regime_label]
        current_regime = int(current_regime_column[~np.isnan(current_regime_column)][-1])

        # future Regime Prediction
        transition_prob = transition_probs[cur_date]

        # run model
        positions = model_init.forward(returns=current_returns,
                                       regimes=current_regime_column,
                                       current_regime=current_regime,
                                       transition_prob=transition_prob)
        
        # store positions
        all_positions.append(pd.DataFrame({pred_date: positions}).T)
    
    all_positions_df = pd.concat(all_positions).fillna(0)
    all_positions_df.columns.name = None
    all_positions_df.index.name = "date"

    return all_positions_df
        
        