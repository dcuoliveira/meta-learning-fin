from tqdm import tqdm
import numpy as np
import pandas as pd

def run_forecasts(returns: pd.DataFrame,
                  regimes: pd.DataFrame,
                  transition_probs: dict,
                  estimation_window: int,
                  model: object,
                  num_assets_to_select: int,
                  fix_start: bool = False):
    
    model_init = model(num_assets_to_select=num_assets_to_select)

    pbar = tqdm(range(0, len(returns) - estimation_window, 1), total=len(returns) - estimation_window)
    for step in pbar:

        if fix_start:
            start = 0
        else:
            start = step

        # subset returns
        current_returns = returns.iloc[start:step + estimation_window, :]
        cur_date = current_returns.index[-1].strftime('%Y-%m-%d')

        # current Regime Identification
        current_regime_column = regimes.iloc[:, step]
        current_regime = int(current_regime_column[~np.isnan(current_regime_column)][-1])

        # future Regime Prediction
        transition_prob = transition_probs[cur_date]
        next_regime = np.argmax(transition_prob[current_regime, :])

        # run model
        pred = model_init.forward(returns=current_returns, current_regime=current_regime, next_regime=next_regime)
        
        