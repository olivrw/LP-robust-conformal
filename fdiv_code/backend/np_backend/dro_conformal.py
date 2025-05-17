import numpy as np
import cvxpy as cp

from sklearn.linear_model import LinearRegression
from sklearn import svm

import np_backend.cython.cython_utils as cython_utils

def dro_conformal_quantile_procedure_cvx(
        training_scores, f_divergence, alpha=0.05, rho=1e-1, epsilon=1e-6, want_bisection=True, solver=cp.ECOS, verbose=False):
    sorted_scores = np.sort(training_scores)
    beta = 1 - alpha
    n = len(training_scores)

    # Initial min and max values
    q_min, q_max = -1, n - 1

    p = cp.Variable(n)
    cns_p_prob_distr = [p >= 0, cp.sum(p) == 1]
    cns_p_f_div = [1.0 / n * cp.sum(f_divergence(n * p)) <= rho]

    # Verifies if there exists p such that D(p||1/n)<= rho
    # and P(S<=scores[(t)]) >= beta
    def solve_q_sub_problem(t, verbose=False):
        obj_t = cp.sum(p[0:(t + 1)])
        prob = cp.Problem(cp.Minimize(obj_t), cns_p_prob_distr + cns_p_f_div)
        try:
            prob.solve(solver=solver, verbose=verbose)
        except:
            print("Solver {} failed for index={}".format(solver, t))
            assert(False)
        return prob.value >= beta, p.value

    # Runs bisection on the index t
    # At most O(log(n)) steps to find the exact solution
    if (want_bisection):
        while q_max > q_min + 1:
            q = int(np.floor((q_min + q_max) / 2.0))
            if verbose:
                print("Solving Subproblem with q={}".format(sorted_scores[q]))
            is_q_big, p_val = solve_q_sub_problem(q, verbose=verbose)
            if is_q_big:
                q_max = q
                worst_case_dist = p_val
            else:
                q_min = q
        try:
            return sorted_scores[q_max], worst_case_dist
        except:
            worst_case_dist = p_val
            return sorted_scores[q_max], worst_case_dist

    else:
        ts = sorted_scores
        for t in ts:
            can_quit, worst_case_dist = solve_q_sub_problem(t)
            if (can_quit):
                return t, worst_case_dist
            
def dro_effective_coverage(beta, rho, f_divergence, solver=cp.MOSEK):
    """
    Returns \phi_f(beta, rho) := 
            min p
            st. f(p/beta) * beta + f(q/(1-beta)) * (1-beta) <= rho
                p + q = 1
    for a sequence of values of beta and rho
    """
    beta_values = np.atleast_1d(beta)
    rho_values = np.atleast_1d(rho)
    
    worst_covering = np.zeros((len(beta_values), len(rho_values)))

    rho_cvx = cp.Parameter(nonneg=True)
    beta_cvx = cp.Parameter(nonneg=True)
    alpha_cvx = cp.Parameter(nonneg=True)

    p_cvx, q_cvx = cp.Variable(), cp.Variable()
    constraints = [
        p_cvx >= 0,
        p_cvx + q_cvx == 1.0,
        beta_cvx * f_divergence(p_cvx / beta_cvx) +
        alpha_cvx * f_divergence(q_cvx / alpha_cvx) <= rho_cvx,
    ]
    cvx_worst_covering_problem = cp.Problem(cp.Minimize(p_cvx), constraints)

    for i, beta in enumerate(beta_values):
        beta_cvx.value = beta
        alpha_cvx.value = 1.0 - beta
        
        for j, rho in enumerate(rho_values):
            rho_cvx.value = rho
            cvx_worst_covering_problem.solve(
                warm_start=True, solver=solver
            )
            worst_covering[i, j] = cvx_worst_covering_problem.value

    return worst_covering

def dro_effective_quantile(beta, rho, f_divergence, solver=cp.MOSEK, use_bisection=True, epsilon=1e-6, lsp_mesh=1000):
    """
    Returns min beta' : \phi_f(beta', rho)>= beta 
    for a sequence of values of beta and rho
    """
    beta_values = np.atleast_1d(beta)
    rho_values = np.atleast_1d(rho)
    
    effective_quantiles = np.zeros((len(beta_values), len(rho_values)))
    
    if use_bisection:
        for j, rho in enumerate(rho_values):
            q_min = np.copy(beta_values)
            q_max = np.full_like(beta_values, np.maximum(1.0-epsilon,beta_values.max()))
            while np.max(q_max - q_min) > epsilon:
                q = (q_min + q_max) / 2.0
                for i, beta in enumerate(beta_values): 
                    threshold = dro_effective_coverage(q[i], rho, f_divergence, solver=solver)
                    if threshold >= beta:
                        q_max[i] = q[i]
                    else:
                        q_min[i] = q[i]
            
            effective_quantiles[:, j] = q
    
    else:
        beta_lsp = np.linspace(beta_values.min(), 0.999, lsp_mesh)
        phi_beta_rho = dro_effective_coverage(beta_lsp, rho_values, f_divergence, solver=solver)

        inverse_phi_index = (
            phi_beta_rho[:, :, np.newaxis] >= beta_values[np.newaxis, np.newaxis, :]
        ).sum(axis=0)
        inverse_phi = np.concatenate([beta_lsp,
                                      [1.0]])[len(beta_lsp) - inverse_phi_index]

        effective_quantiles = inverse_phi.T
        
    return effective_quantiles
    

def dro_conformal_effective_quantile(
    training_scores,
    f_divergence,
    alpha=0.05,
    rho=1e-1,
    effective_quantiles=None,
    solver=cp.MOSEK,
    use_bisection=True,
    lsp_mesh=1000,
    verbose=False
):
    """
    Returns max Quantile(P, 1-alpha)
        such that D_f(P || P_n) <= rho
        where P_n = Multinomial(S_i, 1/n)
        
    This is also Quantile(P_n, Q_eff(1-alpha, rho))
    where Q_eff(1-alpha, rho) is an effective quantile
        that can be directly provided for efficiency
        
    Otherwise the function computes the effective quantile by solving 
    O(log(epsilon)) convex problems
    """
    if effective_quantiles is None:
        effective_quantiles = dro_effective_quantile(
            1.0 - alpha, rho, f_divergence, solver=solver,
            use_bisection=use_bisection, lsp_mesh=lsp_mesh
        )

    return np.quantile(training_scores, effective_quantiles.flatten(),
                      interpolation="higher").reshape(effective_quantiles.shape).squeeze()


def find_worst_coverage(binary_coverage, delta, use_cython=True):
    if use_cython:
#         return cython_utils.find_worst_coverage(binary_coverage.astype(int), delta)
        begin, end = cython_utils.MaximumDensitySegment(
            1.0 - binary_coverage, np.ones_like(binary_coverage, dtype=float),
            float(delta * len(binary_coverage))
        )
        return binary_coverage[begin:end+1].mean()
    
    else:
        n_samples = len(binary_coverage)
        min_sample_size = int(np.ceil(delta * n_samples))
        cumulative_coverage = np.concatenate(
            [[0], binary_coverage.cumsum()]
        )
        minimum_coverage = 1.0
        for i in range(n_samples-min_sample_size+1):
            coverage_starting_from_i = (
                cumulative_coverage[i + min_sample_size:] -
                cumulative_coverage[i]
            )
            coverage_starting_from_i = coverage_starting_from_i / (
                min_sample_size + np.arange(len(coverage_starting_from_i)))

            length = np.argmin(coverage_starting_from_i)
            if coverage_starting_from_i[length] < minimum_coverage:
                minimum_coverage = coverage_starting_from_i[length]        
        return minimum_coverage

def find_worst_case_slab_quantile(direction, features, scores, alpha=0.05, delta=0.2, use_cython=True, tol=1e-4):
    projected_features = features @ direction
    n_samples = features.shape[0]

    direction_sorted_scores = scores[np.argsort(projected_features)]
    t_min = direction_sorted_scores.min()
    t_max = direction_sorted_scores.max()

    t_current = (t_min + t_max) / 2
    while (t_max - t_min) > tol:
        if find_worst_coverage((direction_sorted_scores <= t_current),
                              delta, use_cython=use_cython) >= 1-alpha:
            t_max = t_current
        else:
            t_min = t_current
        t_current = (t_max + t_min) / 2
    return t_max

def find_rho_for_quantile(
    scores, 
    robust_quantile, 
    f_divergence, 
    alpha=0.05,
    delta=0.2,
    solver=cp.MOSEK,
    use_bisection=True,
    tol=1e-5,
    lsp_mesh=1000,
    verbose=False):
    
    rho_min = 0.0
    rho_max = (f_divergence(1.0/delta) + delta* f_divergence(0)).value
    
    rho = (rho_min + rho_max) / 2.0
    
    while (rho_max - rho_min > tol):
        q_rho = dro_conformal_effective_quantile(
            scores,
            f_divergence,
            alpha=alpha,
            rho=rho,
            solver=solver,
            use_bisection=use_bisection,
            verbose=verbose
        )
        if q_rho >= robust_quantile:
            rho_max = rho
        else:
            rho_min = rho
        rho = (rho_min + rho_max) / 2.0
        
    return rho_max

def learnable_direction_quantile(
    data_features, data_scores, calib_ind, model="OLS", alpha=0.05, delta=0.2, verbose=False):
    
    len1=np.int(np.size(calib_ind)/2)  # length of each of the new splits to be formed
    calib_ind1=np.random.choice(calib_ind,size=len1)  # 1st split : indices to train scores on covariates
    calib_ind2=np.setdiff1d(calib_ind,calib_ind1)      # 2nd split : indices to get the worst effective quantile to ensure required coverage on left out set
    X1=data_features[calib_ind1,:]   
    X2=data_features[calib_ind2,:]
    
    S1=data_scores[calib_ind1]  # estimated scores on 1st split
    S2=data_scores[calib_ind2]  # estimated scores on 2nd split
    
    if model=="OLS":
        model1 = LinearRegression()
        model1 = model1.fit(X1, S1)  #learning the direction with OLS
        direction = model1.coef_
        
    elif model=="SVM":
        clf = svm.LinearSVC() # Linear Kernel
        Z=np.zeros(np.size(S1))
        Z[np.where(S1>=np.quantile(S1,1-delta))]=1
        Z[np.where(S1<np.quantile(S1,1-delta))]=-1
        clf.fit(X1, Z)
        direction = clf.coef_[0]
    
    else:
        raise Exception("Model not implemented")
    
    if verbose:
        print(direction)
    return find_worst_case_slab_quantile(direction, X2, S2, alpha=alpha, delta=delta)
