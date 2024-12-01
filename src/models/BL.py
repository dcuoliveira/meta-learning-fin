import torch
import numpy as np
import scipy.optimize as opt
import pandas as pd

from estimators.Estimators import Estimators
from models.Naive import Naive

class BL(Estimators, Naive):
    def __init__(self,
                 strategy_type,
                 risk_aversion: float = 1,
                 tau: float = 0.025,
                 **kwargs) -> None:
        """Black-Litterman model implementation

        Args:
            risk_aversion (float): risk aversion parameter. Defaults to 1. 
                                   Controls the trade-off between risk and return.
            tau (float): scale factor for the uncertainty in the prior estimate of returns.
        """
        super().__init__()
        
        self.strategy_type = strategy_type
        self.risk_aversion = risk_aversion
        self.tau = tau

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool = True) -> torch.Tensor:
        
        c = -1 if maximize else 1
        return (np.dot(weights, self.mean_t) - ((self.risk_aversion / 2) * np.sqrt(np.dot(weights, np.dot(self.cov_t, weights))))) * c

    def compute_mu_bl(self, prior_mean, prior_cov, P, Q, Omega):
        """
        Computes the Black-Litterman expected returns.
        
        Args:
            prior_mean (np.array): Prior mean (market equilibrium returns).
            prior_cov (np.array): Prior covariance matrix.
            P (np.array): Matrix representing views on assets.
            Q (np.array): Views on returns for each asset.
            Omega (np.array): Uncertainty in the views (diagonal matrix).
            
        Returns:
            np.array: Black-Litterman adjusted expected returns.
        """
        if (Q == 0).sum() != Q.shape[0]:
            # Convert all inputs to Float tensors to ensure consistency
            prior_mean = torch.tensor(prior_mean, dtype=torch.float32)
            prior_cov = torch.tensor(prior_cov, dtype=torch.float32)
            P = torch.tensor(P, dtype=torch.float32)
            Q = torch.tensor(Q, dtype=torch.float32)
            Omega = torch.tensor(Omega, dtype=torch.float32)

            tau_prior_cov = self.tau * prior_cov
            middle_term = torch.linalg.inv(tau_prior_cov + P.T @ torch.linalg.inv(Omega) @ P)
            posterior_mean = middle_term @ (tau_prior_cov @ torch.linalg.inv(tau_prior_cov) @ prior_mean + P.T @ torch.linalg.inv(Omega) @ Q)
        else:
            posterior_mean = prior_mean

        return posterior_mean
    
    def forward(self,
                returns: torch.Tensor,
                regimes: pd.DataFrame,
                transition_prob: np.ndarray,
                regime_prob: np.ndarray,
                num_timesteps_out: int = 1,
                random_regime: bool = False,
                **kwargs) -> torch.Tensor:
        
        K = returns.shape[1]

        views, _ = self.compute_forecasts(returns=returns,
                                          regimes=regimes,
                                          regime_prob=regime_prob,
                                          transition_prob=transition_prob,
                                          risk_adjusted=False,
                                          random_regime=random_regime)
        
        # adjust side of positions
        if self.strategy_type == 'lo':
            views = views[views>0]
            if views.shape[0] == 0:
                views = returns.iloc[-1]*0
        elif self.strategy_type == 'lns':
            pass
        else:
            raise Exception(f"Strategy Type Not Supported: {self.strategy_type}")

        # build views matrix Q
        Q = views
        Q_expanded = views.reindex(list(returns.columns), fill_value=0)

        # build matrix of linear combination of the prior expected returns estimate (eq. returns)
        P = torch.zeros(size=(Q.shape[0], K))
        for i in range(P.shape[0]):
            if Q_expanded.iloc[i] != 0:
                if self.strategy_type == 'lo':
                    P_i = torch.zeros((1, K))
                    P_i[0, i] = 1 
                elif self.strategy_type == 'lns':
                    P_i = torch.zeros((1, K))
                    P_i[0, i] = np.sign(Q_expanded[i]) 
                else:
                    raise Exception(f"Strategy Type Not Supported: {self.strategy_type}")

                # add to P matrix
                P[i,:] = P_i

        # build matrix of views uncertainty - hard coded to 5%
        diagonal_tensor = torch.full((P.shape[0],), 0.05)
        Omega = torch.diag_embed(diagonal_tensor)
        
        # transform series to tensor
        returns_tensor = torch.tensor(returns.values)
        Q_tensor = torch.tensor(Q.values)

        # Calculate prior estimates
        self.mean_t = self.MLEMean(returns_tensor)  # Market equilibrium returns
        self.cov_t = self.MLECovariance(returns_tensor)  # Market covariance matrix

        # Calculate Black-Litterman expected returns
        mean_bl_t = self.compute_mu_bl(prior_mean=self.mean_t, prior_cov=self.cov_t, P=P, Q=Q_tensor, Omega=Omega)
        self.mean_t = mean_bl_t  # Update mean_t to BL-adjusted returns

        # Optimization setup
        if self.strategy_type == 'lo':
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}]
            bounds = [(0, 1) for _ in range(K)]
            w0 = np.random.uniform(0, 1, size=K)
        
        elif self.strategy_type == 'lns':
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # "market-neutral" portfolio
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}
            ]
            bounds = [(-1, 1) for _ in range(K)]
            w0 = np.random.uniform(-1, 1, size=K)
        
        else:
            raise Exception(f'Strategy Type not Implemented: {self.strategy_type}')

        # Perform the optimization
        opt_output = opt.minimize(self.objective, w0, constraints=constraints, bounds=bounds)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        # Format output as a DataFrame
        wt = pd.DataFrame(wt, columns=list(returns.columns))
        wt = wt.squeeze(axis=0)

        return wt
