from tqdm import tqdm

from models.ConvictionAdjustedMVO import ConvictionAdjustedMVO

def run_forecasts(data,
                  regimes,
                  regimes_prob,
                  transition_prob,
                  estimation_window,
                  portofolio_method):

    pbar = tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window)
    for step in pbar:
        pass