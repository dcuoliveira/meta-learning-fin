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
        # TODO - what should we do when we have more than one reigme with the same probability?
        probable_next_regime = np.argmax(transition_prob[current_regime, :])

        # run model
        positions = model_init.forward(returns=current_returns,
                                       regimes=current_regime_column,
                                       current_regime=current_regime,
                                       next_regime=probable_next_regime)
        
        # store positions
        all_positions.append(pd.DataFrame(positions, columns=[pred_date]).T)
    
    all_positions_df = pd.concat(all_positions).fillna(0)

    return all_positions_df
        
        