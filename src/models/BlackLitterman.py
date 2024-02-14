import numpy as np
import scipy.optimize as opt
import torch

from estimators.Estimators import Estimators

class BlackLitterman(Estimators):
    def __init__(self,
                 market_prior: np.ndarray,
                 risk_aversion: float,
                 risk_free_rate: float,
                 tau: float,
                 views: np.ndarray,
                 conviction_levels: np.ndarray,
                 P: np.ndarray,
                 Omega: np.ndarray) -> None:
        """
        Implements the Black-Litterman model with conviction-adjusted views.

        Args:
            market_prior (np.ndarray): The market equilibrium expected returns.
            risk_aversion (float): The risk aversion coefficient.
            risk_free_rate (float): The risk-free rate.
            tau (float): A scalar indicating the uncertainty of the prior.
            views (np.ndarray): An array of the investor's views on the returns of the assets.
            conviction_levels (np.ndarray): An array of conviction levels for each view.
            P (np.ndarray): A matrix that identifies the assets involved in the views.
            Omega (np.ndarray): A diagonal matrix of the variances of the views.
        """
        super().__init__()
        
        self.market_prior = market_prior
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.views = views
        self.conviction_levels = conviction_levels
        self.P = P
        self.Omega = Omega

        # Calculate the adjusted views based on conviction levels
        self.adjusted_views = self.views * self.conviction_levels

    def calculate_posterior_returns(self, sigma: np.ndarray) -> np.ndarray:
        """
        Calculates the posterior expected returns using the Black-Litterman formula.

        Args:
            sigma (np.ndarray): The covariance matrix of asset returns.

        Returns:
            np.ndarray: The posterior expected returns.
        """
        # Calculate the reverse optimization to find the implied equilibrium returns
        pi = self.risk_aversion * np.dot(sigma, self.market_prior)

        # Adjust Omega based on conviction levels, if necessary
        adjusted_Omega = self.Omega * self.conviction_levels

        # Black-Litterman formula to combine the market equilibrium returns with the investor's views
        M = np.linalg.inv(np.linalg.inv(self.tau * sigma) + np.dot(np.dot(self.P.T, np.linalg.inv(adjusted_Omega)), self.P))
        adjusted_returns = np.dot(M, np.dot(np.linalg.inv(self.tau * sigma), pi) + np.dot(np.dot(self.P.T, np.linalg.inv(adjusted_Omega)), self.adjusted_views))
        
        return adjusted_returns

    def objective(self, weights: np.ndarray, returns: np.ndarray, sigma: np.ndarray) -> float:
        """
        The objective function to be minimized.

        Args:
            weights (np.ndarray): Portfolio weights.
            returns (np.ndarray): Expected returns.
            sigma (np.ndarray): Covariance matrix.

        Returns:
            float: The negative of the portfolio's expected utility.
        """
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_volatility**2
        return -utility

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:
        """
        Optimizes the portfolio based on the posterior returns.

        Args:
            sigma (np.ndarray): The covariance matrix of asset returns.
            num_assets (int): The number of assets.

        Returns:
            np.ndarray: The optimized portfolio weights.
        """
        K = returns.shape[1]

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

        # Calculate posterior expected returns
        self.posterior_returns = self.calculate_posterior_returns(self.cov_t)

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

        # Optimization
        opt_output = opt.minimize(self.objective, w0, method='SLSQP', constraints=constraints, bounds=bounds)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt