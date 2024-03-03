import pandas as pd

from ModelUtils import ModelUtils as mu

class Naive:
    def __init__(self, num_assets_to_select: int):
        self.num_assets_to_select = num_assets_to_select

    def forward(self, returns: pd.DataFrame, current_regime: int, next_regime: int):
        return data.mean().nlargest(self.num_assets_to_select).index.tolist()