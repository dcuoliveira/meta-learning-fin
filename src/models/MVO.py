import torch
import numpy as np
import scipy.optimize as opt
import pandas as pd

from estimators.Estimators import Estimators

class MVO(Estimators):
    def __init__(self,
                 strategy_type,
                 risk_aversion: float=1,
                 mean_estimator: str="mle",
                 covariance_estimator: str="mle",
                 **kwargs) -> None:
        """"
        This function impements the mean-variance optimization (MVO) method proposed by Markowitz (1952).

        Args:
            risk_aversion (float): risk aversion parameter. Defaults to 0.5. 
                                   The risk aversion parameter is a scalar that controls the trade-off between risk and return.
                                   According to Ang (2014), the risk aversion parameter of a risk neutral individual ranges from 1 and 10.
            mean_estimator (str): mean estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.

        References:
        Markowitz, H. (1952) Portfolio Selection. The Journal of Finance.
        Ang, Andrew, (2014). Asset Management: A Systematic Approach to Factor Investing. Oxford University Press. 
        """
        super().__init__()
        
        self.strategy_type = strategy_type
        self.risk_aversion = risk_aversion
        self.mean_estimator = mean_estimator
        self.covariance_estimator = covariance_estimator
        self.estimated_means = list()
        self.estimated_covs = list()

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool=True) -> torch.Tensor:
        
        c = -1 if maximize else 1
        
        return (np.dot(weights, self.mean_t) - ((self.risk_aversion/2) * np.sqrt(np.dot(weights, np.dot(self.cov_t, weights))) )) * c

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int=1,
                **kwargs) -> torch.Tensor:
        
        returns_tensor = torch.tensor(returns.values)
        
        K = returns.shape[1]

        # expected return estimates
        self.mean_t = self.MLEMean(returns_tensor)

        # covariance estimates
        self.cov_t = self.MLECovariance(returns_tensor)

        if self.strategy_type == 'lo':
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1} # allocate exactly all your assets (no leverage)
            ]
            bounds = [(0, 1) for _ in range(K)] # long-only

            w0 = np.random.uniform(0, 1, size=K)
        elif self.strategy_type == 'lns':
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # "market-neutral" portfolio
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}, # allocate exactly all your assets (no leverage)
            ]
            bounds = [(-1, 1) for _ in range(K)]

            w0 = np.random.uniform(-1, 1, size=K)
        else:
            raise Exception(f'Strategy Type not Implemented: {self.strategy_type}')

        # perform the optimization
        opt_output = opt.minimize(self.objective, w0, constraints=constraints, bounds=bounds)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        wt = pd.DataFrame(wt, columns=list(returns.columns))
        wt = wt.squeeze(axis=0)

        return wt
