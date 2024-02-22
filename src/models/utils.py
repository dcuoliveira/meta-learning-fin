from models.ConvictionAdjustedMVO import ConvictionAdjustedMVO
from models.Naive import Naive

def parse_portfolio_method(portfolio_method: str):
    if portfolio_method == "naive":
        return Naive
    elif portfolio_method == "conviction_mvo":
        return ConvictionAdjustedMVO
    else:
        raise ValueError(f"Invalid portfolio method: {portfolio_method}")