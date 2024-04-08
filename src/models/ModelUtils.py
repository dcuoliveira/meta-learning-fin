from models.ConvictionAdjustedMVO import ConvictionAdjustedMVO
from models.Naive import Naive
from models.WeightedNaive import WeightedNaive

class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def parse_portfolio_method(portfolio_method: str):
        if portfolio_method == "naive":
            return Naive
        elif portfolio_method == "weighted-naive":
            return WeightedNaive
        elif portfolio_method == "conviction_mvo":
            return ConvictionAdjustedMVO
        else:
            raise ValueError(f"Invalid portfolio method: {portfolio_method}")
    
    def parse_regimes(self):
        pass