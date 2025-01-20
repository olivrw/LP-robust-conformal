import numpy as np


class FDivRobustCP:
    def __init__(self, rho, tol, f, is_chisq):
        self.rho = rho
        self.tol = tol
        self.f = f
        self.is_chisq = is_chisq

    def g(self, z, beta):
        return beta*self.f(z/beta) + (1-beta)*self.f((1-z)/(1-beta))

    def g_f_rho(self, beta):
        if self.is_chisq is True:
            ans = np.maximum(0, beta-np.sqrt(2.*self.rho*beta*(1-beta)))
        else:
            left, right = 0.0, 1.0
            while right-left > self.tol:
                mid = (left+right)/2.
                if self.g(mid, beta) <= self.rho:
                    right = mid
                else:
                    left = mid
            ans = (left+right)/2.
        return ans

    def g_f_rho_inv(self, tau):
        left, right = 0.0, 1.0
        while right-left > self.tol:
            beta = (left+right)/2.
            z = self.g_f_rho(beta)
            if z <= tau:
                left = beta
            else:
                right = beta
        return (left+right)/2.

    def adjusted_level(self, n, alpha):
        return 1 - self.g_f_rho((1+(1/n))*self.g_f_rho_inv(1-alpha))

    def adjusted_quantile(self, calib_scores, n, alpha):
        q_level = 1.-self.adjusted_level(n, alpha)
        return np.quantile(calib_scores, q_level, method='higher')


if __name__ == '__main__':
    def f(u):
        return (u-1)**2

    # Initialize the class
    rho = 5  # Example value
    tol = 1e-12
    cp = FDivRobustCP(rho, tol, f)

    # Test g_f_rho method
    beta = 0.9  # Example value
    z_min = cp.g_f_rho(beta)
    print(f"Minimum z for beta={beta}: {z_min}")
    print(f"function at z_min={cp.g(z_min, beta)}")

    # Test g_f_rho_inv method
    tau = 0.3  # Example value
    beta_max = cp.g_f_rho_inv(tau)
    print(f"Maximum beta for tau={tau}: {beta_max}")
    print(f"inverse function val at beta_max={cp.g_f_rho(beta_max)}")
