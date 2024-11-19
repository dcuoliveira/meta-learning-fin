from models.Naive import Naive
from models.WeightedNaive import WeightedNaive
from models.LinearModels import LinearModels
from models.MVO import MVO
from models.BL import BL

class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def parse_portfolio_method(portfolio_method: str):
        if portfolio_method == "naive":
            return Naive
        elif portfolio_method == "weighted-naive":
            return WeightedNaive
        elif portfolio_method.startswith("linear"):
            return LinearModels
        elif portfolio_method.startswith("mvo"):
            return MVO
        elif portfolio_method.startswith("bl"):
            return BL
        else:
            raise ValueError(f"Invalid portfolio method: {portfolio_method}")
    
    def parse_regimes(self):
        pass