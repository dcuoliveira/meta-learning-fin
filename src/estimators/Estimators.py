import torch

class Estimators:
    """
    This class implements the estimators for all the unknown quantites we have on the optimization problems.

    """
    def __init__(self) -> None:
        pass

    def MLEMean(self,
                returns: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the mean of the returns.

        Args:
            returns (torch.tensor): returns tensor.
        
        Returns:
            mean_t (torch.tensor): MLE estimates for the mean of the returns.
        """
        mean_t = torch.mean(returns, axis = 0)

        return mean_t
    
    def MLECovariance(self,
                      returns: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the covariance of the returns.

        Args:
            returns (torch.tensor): returns tensor.

        Returns:
            cov_t (torch.tensor): MLE estimates for the covariance of the returns.
        """
        
        cov_t = torch.cov(returns.T,correction = 0)

        return cov_t
    
    def MLEUncertainty(self,
                       T: float,
                       cov_t: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the uncertainty of the returns estimates.
        This method is used for the Robust Portfolio Optimization problem.

        Args:
            T (float): number of time steps.
            cov_t (torch.tensor): covariance tensor.

        Returns:
            omega_t (torch.tensor): MLE estimates for the uncertainty of the returns estimates.
        """
        
        omega_t = torch.zeros_like(cov_t)
        cov_t_diag = torch.diagonal(cov_t, 0)/T
        omega_t.diagonal().copy_(cov_t_diag)

        return omega_t