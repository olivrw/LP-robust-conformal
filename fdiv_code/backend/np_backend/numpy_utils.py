import numpy as np
from scipy.stats import norm, ncx2

def compute_confidence_sets(alpha, probas, U=0):
    """
    Computes for each row the smallest (potentially randomized set)
    such that P(set) >= 1 - alpha
    """
    N, K = probas.shape
    argsort_probas = np.argsort(probas, axis=1)[:,::-1]
    sorted_probas  =  np.sort(probas)[:, ::-1]
    confidenceSet = np.full((N,K), True)
    
    confidenceSet[:,1:] = np.logical_not(
       sorted_probas.cumsum(axis=1) >= 1 - alpha
    )[:,:-1]
    confidenceSize = confidenceSet.sum(axis=1)
    
    V = 1.0 / sorted_probas[np.arange(N), confidenceSize-1] * (
        (sorted_probas.cumsum(axis=1)[
            np.arange(N), confidenceSize - 1
        ]) - (1-alpha)
    )
    
    confidenceSet[np.arange(N),confidenceSize-1] = (U > V)
    confidenceSize =  confidenceSet.sum(axis=1)
    
    confidenceSetUnsorted = np.full((N,K), False)
    confidenceSetUnsorted[
        np.repeat(np.arange(N)[:, np.newaxis], K, axis=1), argsort_probas
    ] = confidenceSet
    return confidenceSize, confidenceSetUnsorted

def compute_sc_romano_confidence_scores(probas, U):
    """ Returns the approximate p-value E(x, y, u, \pi)
    as in Emmanuel's Paper
    """
    
    alphas = np.linspace(0.0, 1.0, 1001)
    n, k = probas.shape
    E_scores = np.ones_like(probas)
    
    indices = np.full(n,True)
    S_cs = np.full((n,k),False)
    
    for alpha in alphas:
        _, S_cs[indices,:] = compute_confidence_sets(alpha, probas[indices,:], U[indices])
        indices = np.any(S_cs, axis=1)
        E_scores[S_cs] = 1 - alpha
    return E_scores


def compute_sc_romano_direct_confidence_scores(probas, U):
    """ Returns the approximate p-value E(x, y, u, \pi)
    as in Emmanuel's Paper
    """
    N, K = probas.shape
    argsort_probas = np.argsort(probas, axis=1)[:,::-1]
    sorted_probas  =  np.sort(probas)[:, ::-1]
    
    E_sorted_scores = np.zeros((N,K))
    E_sorted_scores += (1.0 - U[:,np.newaxis]) * sorted_probas
    E_sorted_scores[:,1:] += sorted_probas.cumsum(axis=1)[:,:-1]
    
    E_scores = np.zeros((N,K))
    E_scores[
        np.repeat(np.arange(N)[:, np.newaxis], K, axis=1), argsort_probas
    ] = E_sorted_scores
    return E_scores

def get_complete_configurations(n):
    if n == 1:
        return np.array([[0], [1]])
    else:
        config = get_complete_configurations(n - 1)
        N = config.shape[0]

        return np.concatenate(
            [
                np.concatenate([np.zeros((N, 1)),
                                np.ones((N, 1))], axis=0),
                np.concatenate([config, config], axis=0)
            ],
            axis=1
        )
    
    
def compute_parwise_kernel(
    X1=None,
    X2=None,
    D=None,
    kernel="gaussian",
    sigma=1.0,
    precomputed=False,
    tol=1e-10,
    return_pairwise_distance=False,
):
    """
    Returns the NxP matrix K(x_i,z_j)
    - X1 = [x_1^T,...,x_N^T] Nxd matrix
    - X2 = [z_1^T,...,z_P^T] Pxd matrix
    - K is a given kernel
    """
    if X2 is None:
        X2 = X1
    else:
        assert X1.shape[1] == X2.shape[1]
    if not (precomputed):
        G = X1 @ X2.T
        N1 = (X1 ** 2).sum(axis=1)
        N2 = (X2 ** 2).sum(axis=1)
        D_square = -2.0 * G + N1[:, np.newaxis] + N2[np.newaxis, :] + tol
    else:
        D_square = D ** 2
    if return_pairwise_distance:
        if kernel == "gaussian":
            return np.exp(-D_square / (2.0 * sigma ** 2)), np.sqrt(D_square)
    else:
        if kernel == "gaussian":
            return np.exp(-D_square / (2.0 * sigma ** 2))
    raise Exception("Kernel Unknown")
    return 0


def estimate_probability_from_kernel(kernel_matrix, y):
    return kernel_matrix / kernel_matrix.sum(axis=1, keepdims=True) @ y


# Pinball gradient function (quantile loss)
def numpy_pinball_loss_gradient(x, alpha=0.05):
    return np.where(x > 0, alpha, -(1 - alpha))

# Pinball function (quantile loss)
def numpy_pinball_loss(x, alpha=0.05):
    return np.where(x > 0, alpha * x, -(1 - alpha) * x)

# Softmax function
def numpy_softmax(z):
    if z.ndim == 1:
        z = z[np.newaxis,:]
    logits = z - z.max(axis=1, keepdims=True)
    probas = np.exp(logits)
    return probas / probas.sum(axis=1, keepdims=True)

# Logistic loss function
# x is an Nxd matrix
# y is a N-dimensional vector of labels
# theta is a d-dimensional vector
def numpy_logistic_loss(x, y, theta):
    return -np.log(select_from_each_row(softmax(x @ theta), y))

# Select specific indices from each row of A
# Returns the vector with elements A[i,indices[i]]
def select_from_each_row(A, indices):
    return A[np.arange(A.shape[0]), indices]

# Entropy function
# pk represents N different discrete probability 
# distributions of support size K and is a NxK matrix
def entropy(pk):
    return -np.sum(pk * np.log(pk), axis=1)


# Projection onto the spectral norm ball
def projectOnSpectralNormBall(Sigma, rho=1.0):
    u, s, vh = np.linalg.svd(Sigma)
    return (u * np.clip(s, -rho, rho)) @ vh

# Returns the vector x_i^T * sigma^-1 * x_i 
# where x = [x_1^T, ..., x_N^T] is a matrix of size N x d
# and sigma is a dxd PSD matrix
def quadOverLin(x, sigma):
    return np.sum((x @ np.linalg.inv(sigma)) * x, axis=1)


def format_binary_labels(y, input_format="binary", output_format="rademacher"):
    if input_format == output_format:
        return y
    elif input_format == "binary" and output_format == "rademacher":
        return 2.0 * y - 1.0
    elif input_format == "rademacher" and output_format == "binary":
        return 1.0 * (y > 0.0)
    else:
        raise NotImplementedError()
    return

def normalizeProbabilities(probas):
    """
    Normalizes probabilities to 1
    Assumes that probas > 0
    Designed for cases in which probas can be very close to 0 
    """
    log_probas = np.log(probas)
    log_probas = log_probas - log_probas.max(axis=1, keepdims=True)
    probas = np.exp(log_probas)
    return probas / probas.sum(axis=1, keepdims=True)

### Functions for predictions with Gaussian Noise Addition


def computeCoverageProbability(t, py_x, pred, eps, sigma=None):
    """
    Computes the following probability
    P(||y+eps*z - pred||^2_sigma_inv <= t)
    where y is drawn from py_x and z~N(0,sigma)
    """
    if py_x.ndim == 1:
        py_x = py_x[np.newaxis, :]
        pred = pred[np.newaxis, :]
    py_x = py_x / py_x.sum(axis=1, keepdims=True)
    N, K = py_x.shape
    nc_params = np.zeros((N, K))
    if sigma is not None:
        for k in range(K):
            ek = np.zeros(K)
            ek[k] = 1.0
            nc_params[:, k] = quadOverLin(ek[np.newaxis, :] - pred, sigma) / eps ** 2
    else:
        for k in range(K):
            ek = np.zeros(K)
            ek[k] = 1.0
            nc_params[:, k] = np.sum((ek[np.newaxis, :] - pred) ** 2, axis=1) / eps ** 2
    print(nc_params, ncx2.cdf(t[:, np.newaxis] / eps ** 2, K, nc_params))
    return np.sum(py_x * ncx2.cdf(t[:, np.newaxis] / eps ** 2, K, nc_params), axis=1)


def estimateQuantile(alpha, py_x, pred, eps, sigma=None, tol=1e-6):
    """
    Computes t_alpha such that
    P(||y+eps*z - pred||^2_sigma_inv <= t_alpha) = 1 - alpha
    where y is drawn from py_x and z~N(0,sigma)
    """
    if py_x.ndim == 1:
        py_x = py_x[np.newaxis, :]
        pred = pred[np.newaxis, :]
    py_x = normalizeProbabilities(py_x)
    N, K = py_x.shape
    nc_params = np.zeros((N, K))
    if sigma is not None:
        for k in range(K):
            ek = np.zeros(K)
            ek[k] = 1.0
            nc_params[:, k] = quadOverLin(ek[np.newaxis, :] - pred, sigma) / eps ** 2
    else:
        for k in range(K):
            ek = np.zeros(K)
            ek[k] = 1.0
            nc_params[:, k] = np.sum((ek[np.newaxis, :] - pred) ** 2, axis=1) / eps ** 2

    tmin = np.zeros(N)
    tmax = np.full(N, 10.0)
    t = 0.5 * tmin + 0.5 * tmax

    while np.max(tmax - tmin) > tol:
        coverage = np.sum(
            py_x * ncx2.cdf(t[:, np.newaxis] / eps ** 2, K, nc_params), axis=1
        )
        tmax = t * (coverage > 1 - alpha) + tmax * (coverage <= 1 - alpha)
        tmin = tmin * (coverage > 1 - alpha) + t * (coverage <= 1 - alpha)
        t = 0.5 * tmin + 0.5 * tmax
    return t


def get_MCProbability_for_misspecified_scenario(beta, t, M_trials=int(1e4)):
    """
    MC Estimate of E(1/(2*exp(t)*cosh(beta*z)+1)) where z ~ N(0,1)
    """
    z = np.random.randn(1,M_trials)
    return np.mean(1.0 / ((np.exp(beta* z) + np.exp(beta* z)) * np.exp(t[:,np.newaxis])+1.0), axis=1)
    
    
