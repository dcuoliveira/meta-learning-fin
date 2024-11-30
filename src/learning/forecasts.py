from tqdm import tqdm
import numpy as np
import pandas as pd

def run_forecasts(returns: pd.DataFrame,
                  features: pd.DataFrame,
                  regimes: pd.DataFrame,
                  regimes_probs: dict,
                  transition_probs: dict,
                  estimation_window: int,
                  model: object,
                  portfolio_method: str,
                  cv_search_type: str,
                  cv_split_type: str,
                  cv_folds: int,
                  cv_iters: int,
                  num_assets_to_select: int,
                  fix_start: bool = False,
                  strategy_type: str = 'm',
                  random_regime: bool = False,):

    model_init = model(num_assets_to_select=num_assets_to_select,
                       strategy_type=strategy_type,
                       portfolio_method=portfolio_method,
                       cv_search_type=cv_search_type,
                       cv_split_type=cv_split_type,
                       cv_folds=cv_folds,
                       cv_iters=cv_iters,)

    pbar = tqdm(range(0, len(returns) - estimation_window, 1), total=len(returns) - estimation_window, desc="Running Forecasts")
    all_positions = []
    for step in pbar:

        if fix_start:
            start = 0
        else:
            start = step

        # subset returns
        train_returns = returns.iloc[start:step + estimation_window, :]
        train_date = train_returns.index[-1].strftime('%Y-%m-%d')
        test_date = returns.index[step + estimation_window].strftime('%Y-%m-%d')

        # subset features
        train_features = features.loc[:train_date]
        test_features = pd.DataFrame(features.iloc[train_features.shape[0], :]).T

        # current Regime Identification
        # find date of last nan value for regimes datafrmae
        regimes_index = regimes.apply(lambda x: x.dropna().index[-1])
        regime_label = regimes_index[regimes_index == train_date].index[0]
        current_regime_column = regimes.loc[:, regime_label]
        current_regime = int(current_regime_column[~np.isnan(current_regime_column)][-1])
        if random_regime:
            current_regime = np.random.choice(list(range(6)))

        # future Regime Prediction
        transition_prob = transition_probs[train_date]
        regime_prob = regimes_probs[train_date]
        if random_regime:
            transition_prob = np.ones((transition_prob.shape[0], transition_prob.shape[1])) * 1/transition_prob.shape[0]
            regime_prob = np.ones((regime_prob.shape[0], regime_prob.shape[1])) * 1/transition_prob.shape[0]

        # run model
        positions = model_init.forward(returns=train_returns,
                                       features=train_features,
                                       test_features=test_features,
                                       regimes=current_regime_column,
                                       current_regime=current_regime,
                                       transition_prob=transition_prob,
                                       regime_prob=regime_prob,
                                       random_regime=random_regime,)

        # store positions
        all_positions.append(pd.DataFrame({test_date: positions}).T)
    all_positions_df = pd.concat(all_positions).fillna(0)
    all_positions_df.columns.name = None
    all_positions_df.index.name = "date"

    return all_positions_df
