import torch
import numpy as np
import scipy.optimize as opt

from estimators.Estimators import Estimators

class ConvictionAdjustedMVO(Estimators):
    def __init__(self,
                 risk_aversion: float=1,
                 mean_estimator: str="mle",
                 covariance_estimator: str="mle") -> None:
        """
        This function implements the Mean-Variance Optimization (MVO) method with conviction-adjusted returns.

        Args:
            risk_aversion (float): Risk aversion parameter.
            mean_estimator (str): Mean estimator to be used.
            covariance_estimator (str): Covariance estimator to be used.

        """
        super().__init__()
        
        self.risk_aversion = risk_aversion
        self.mean_estimator = mean_estimator
        self.covariance_estimator = covariance_estimator
        self.estimated_means = list()
        self.estimated_covs = list()

    def adjust_returns_with_conviction(self, mean_returns: np.ndarray) -> np.ndarray:
        """
        Adjusts the mean returns based on the conviction levels.

        Args:
            mean_returns (np.ndarray): The estimated mean returns for each asset.

        Returns:
            np.ndarray: The conviction-adjusted mean returns.
        """
        if self.conviction_levels is None:
            return mean_returns
        else:
            # Adjust the mean returns by the conviction levels
            adjusted_returns = mean_returns * self.conviction_levels
            return adjusted_returns

    def objective(self, weights: torch.Tensor, maximize: bool=True) -> torch.Tensor:
        c = -1 if maximize else 1
        
        adjusted_mean_returns = self.adjust_returns_with_conviction(self.mean_t)
        return (np.dot(weights, adjusted_mean_returns) - ((self.risk_aversion / 2) * np.sqrt(np.dot(weights, np.dot(self.cov_t, weights))))) * c

    def forward(self,
                returns: torch.Tensor,
                conviction_levels: np.ndarray,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:
        
        K = returns.shape[1]
        self.conviction_levels = conviction_levels

        # mean estimator
        if self.mean_estimator == "mle":
            self.mean_t = self.MLEMean(returns)
        elif (self.mean_estimator == "cbb") or (self.mean_estimator == "nobb") or (self.mean_estimator == "sb"):
            self.mean_t = self.DependentBootstrapMean(returns=returns,
                                                      boot_method=self.mean_estimator,
                                                      Bsize=50,
                                                      rep=1000)
        elif self.mean_estimator == "rbb":
            self.mean_t = self.DependentBootstrapMean(returns=returns,
                                                    boot_method=self.mean_estimator,
                                                    Bsize=50,
                                                    rep=1000,
                                                    max_p=4)
        else:
            raise NotImplementedError
        self.estimated_means.append(self.mean_t[None, :])

        # covariance estimator
        if self.covariance_estimator == "mle":
            self.cov_t = self.MLECovariance(returns)
        elif (self.mean_estimator == "cbb") or (self.mean_estimator == "nobb") or (self.mean_estimator == "sb"):
            self.cov_t = self.DependentBootstrapCovariance(returns=returns,
                                                           boot_method=self.covariance_estimator,
                                                           Bsize=50,
                                                           rep=1000)
        elif self.covariance_estimator == "rbb":
            self.cov_t = self.DepenBootstrapCovariance(returns=returns,
                                                       boot_method=self.covariance_estimator,
                                                       Bsize= 50,
                                                       rep=1000,
                                                       max_p= 4)
        else:
            raise NotImplementedError
        self.estimated_covs.append(self.cov_t)

        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the weights sum to one
            ]
            bounds = [(0, 1) for _ in range(K)]

            w0 = np.random.uniform(0, 1, size=K)
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # the weights sum to zero
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1},  # the weights sum to zero
            ]
            bounds = [(-1, 1) for _ in range(K)]

            w0 = np.random.uniform(-1, 1, size=K)

        # perform the optimization
        opt_output = opt.minimize(self.objective, w0, constraints=constraints, bounds=bounds)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt