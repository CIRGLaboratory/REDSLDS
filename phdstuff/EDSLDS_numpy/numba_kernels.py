"""
Numba-accelerated kernels for EDSLDS performance optimization.

This module provides JIT-compiled versions of critical inner loops and
mathematical operations that benefit from compiled code execution.

Usage:
    from .numba_kernels import (
        backward_kalman_loop,
        forward_kalman_loop,
        sigmoid_numba,
        mahalanobis_batch,
        cholesky_mvn_sample
    )

The functions gracefully fall back to pure NumPy if Numba is not available.
"""

import os

import numpy as np

# Try to import Numba, provide fallback flag if not available
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    # The parallel kernels below run inside forked pathos workers. Numba's
    # default OpenMP (libgomp) layer aborts those workers ("fork() called from
    # a process already using GNU OpenMP") and pool.map never returns; the
    # fork-safe workqueue layer is the default here unless the env overrides.
    numba.config.THREADING_LAYER = os.environ.get("NUMBA_THREADING_LAYER", "workqueue")
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# =============================================================================
# BASIC MATH OPERATIONS
# =============================================================================

@njit(cache=True)
def sigmoid_numba(x):
    """Numerically stable sigmoid function - Numba accelerated.

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Sigmoid(x) computed in a numerically stable way
    """
    result = np.empty_like(x)
    for i in range(x.size):
        xi = x.flat[i]
        if xi >= 0:
            result.flat[i] = 1.0 / (1.0 + np.exp(-xi))
        else:
            exp_xi = np.exp(xi)
            result.flat[i] = exp_xi / (1.0 + exp_xi)
    return result


# =============================================================================
# MATRIX OPERATIONS
# =============================================================================

@njit(cache=True)
def invert_2d_numba(A):
    """Invert a 2D matrix using NumPy's linear algebra.

    Falls back to pseudo-inverse behavior for singular matrices.
    """
    return np.linalg.inv(A)


@njit(cache=True)
def mahalanobis_distance_single(diff, Sigma_inv):
    """Compute Mahalanobis distance for a single observation.

    Parameters
    ----------
    diff : np.ndarray (D,)
        Difference vector (x - mu)
    Sigma_inv : np.ndarray (D, D)
        Inverse covariance matrix

    Returns
    -------
    float
        Mahalanobis distance squared: diff.T @ Sigma_inv @ diff
    """
    D = diff.shape[0]
    result = 0.0
    for i in range(D):
        for j in range(D):
            result += diff[i] * Sigma_inv[i, j] * diff[j]
    return result


@njit(cache=True, parallel=True)
def mahalanobis_batch(diff, Sigma_inv):
    """Compute Mahalanobis distances for batch of observations.

    Parameters
    ----------
    diff : np.ndarray (T, D)
        Difference matrix where each row is (x_t - mu)
    Sigma_inv : np.ndarray (D, D)
        Inverse covariance matrix

    Returns
    -------
    np.ndarray (T,)
        Mahalanobis distances squared for each observation
    """
    T, D = diff.shape
    result = np.empty(T)
    for t in prange(T):
        val = 0.0
        for i in range(D):
            for j in range(D):
                val += diff[t, i] * Sigma_inv[i, j] * diff[t, j]
        result[t] = val
    return result


@njit(cache=True, parallel=True)
def mahalanobis_batch_multi_sigma(diff, Sigma_invs):
    """Compute Mahalanobis distances for batch with state-dependent covariances.

    Parameters
    ----------
    diff : np.ndarray (K, T, D)
        Difference matrix for each state k
    Sigma_invs : np.ndarray (K, D, D)
        Inverse covariance matrices for each state

    Returns
    -------
    np.ndarray (K, T)
        Mahalanobis distances squared for each state and time
    """
    K, T, D = diff.shape
    result = np.empty((K, T))
    for k in prange(K):
        for t in range(T):
            val = 0.0
            for i in range(D):
                for j in range(D):
                    val += diff[k, t, i] * Sigma_invs[k, i, j] * diff[k, t, j]
            result[k, t] = val
    return result


# =============================================================================
# MULTIVARIATE NORMAL SAMPLING (Cholesky-based)
# =============================================================================

@njit(cache=True)
def cholesky_mvn_sample(mu, L, z):
    """Sample from multivariate normal using Cholesky decomposition.

    x = mu + L @ z where z ~ N(0, I) and L = cholesky(Sigma)

    Parameters
    ----------
    mu : np.ndarray (D,)
        Mean vector
    L : np.ndarray (D, D)
        Lower Cholesky factor of covariance matrix
    z : np.ndarray (D,)
        Standard normal samples

    Returns
    -------
    np.ndarray (D,)
        Sample from N(mu, L @ L.T)
    """
    D = mu.shape[0]
    result = mu.copy()
    for i in range(D):
        for j in range(i + 1):
            result[i] += L[i, j] * z[j]
    return result


@njit(cache=True)
def cholesky_lower(A):
    """Compute lower Cholesky decomposition L such that A = L @ L.T

    Parameters
    ----------
    A : np.ndarray (D, D)
        Positive definite matrix

    Returns
    -------
    np.ndarray (D, D)
        Lower triangular Cholesky factor
    """
    D = A.shape[0]
    L = np.zeros((D, D))

    for i in range(D):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]

            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    val = 1e-10  # Numerical stability
                L[i, j] = np.sqrt(val)
            else:
                if L[j, j] > 0:
                    L[i, j] = (A[i, j] - s) / L[j, j]
                else:
                    L[i, j] = 0.0
    return L


# =============================================================================
# KALMAN FILTER CORE LOOPS
# =============================================================================

@njit(cache=True)
def backward_kalman_step(Lambda_next, Theta_next, dSigma_inv, D_t, C_t, R_inv, obs_t):
    """Single backward Kalman filter step.

    Parameters
    ----------
    Lambda_next : np.ndarray (D, D)
        Information matrix from t+1
    Theta_next : np.ndarray (D,)
        Information vector from t+1
    dSigma_inv : np.ndarray (D, D)
        Inverse dynamics covariance
    D_t : np.ndarray (D, D)
        Dynamics transition matrix
    C_t : np.ndarray (obs_dim, D)
        Emission matrix
    R_inv : np.ndarray (obs_dim, obs_dim)
        Inverse emission covariance
    obs_t : np.ndarray (obs_dim,)
        Observation at time t

    Returns
    -------
    Lambda_t : np.ndarray (D, D)
        Updated information matrix
    Theta_t : np.ndarray (D,)
        Updated information vector
    """
    D = Lambda_next.shape[0]

    # Compute J = Lambda_next @ inv(Lambda_next + dSigma_inv)
    C1 = Lambda_next + dSigma_inv
    C1_inv = np.linalg.inv(C1)
    J = Lambda_next @ C1_inv

    # L = I - J
    L = np.eye(D) - J

    # Predict step
    temp = L @ Lambda_next @ L.T + J @ dSigma_inv @ J.T
    Lambda_1 = D_t.T @ temp @ D_t
    Theta_1 = D_t.T @ L @ Theta_next

    # Update step with observation
    Lambda_t = Lambda_1 + C_t.T @ R_inv @ C_t
    Theta_t = Theta_1 + C_t.T @ R_inv @ obs_t

    return Lambda_t, Theta_t


@njit(cache=True)
def backward_kalman_loop(T, Lambdas, Thetas, dSigma_invs, Ds, Cs, R_invs, obs):
    """Full backward Kalman filter loop.

    Parameters
    ----------
    T : int
        Number of time steps
    Lambdas : np.ndarray (T, D, D)
        Output information matrices
    Thetas : np.ndarray (T, D)
        Output information vectors
    dSigma_invs : np.ndarray (T, D, D)
        Pre-inverted dynamics covariances
    Ds : np.ndarray (T, D, D)
        Dynamics transition matrices
    Cs : np.ndarray (T, obs_dim, D)
        Emission matrices
    R_invs : np.ndarray (T, obs_dim, obs_dim)
        Pre-inverted emission covariances
    obs : np.ndarray (T, obs_dim)
        Observations
    """
    D = Lambdas.shape[1]

    # Initialize last timestep
    Lambdas[T - 1] = Cs[T - 1].T @ R_invs[T - 1] @ Cs[T - 1]
    Thetas[T - 1] = Cs[T - 1].T @ R_invs[T - 1] @ obs[T - 1]

    # Backward loop
    for t in range(T - 2, -1, -1):
        # Compute step
        C1 = Lambdas[t + 1] + dSigma_invs[t + 1]
        C1_inv = np.linalg.inv(C1)
        J = Lambdas[t + 1] @ C1_inv
        L = np.eye(D) - J

        # Predict
        temp = L @ Lambdas[t + 1] @ L.T + J @ dSigma_invs[t + 1] @ J.T
        Lambda_1 = Ds[t + 1].T @ temp @ Ds[t + 1]
        Theta_1 = Ds[t + 1].T @ L @ Thetas[t + 1]

        # Update (for t > 0)
        if t > 0:
            Lambdas[t] = Lambda_1 + Cs[t].T @ R_invs[t] @ Cs[t]
            Thetas[t] = Theta_1 + Cs[t].T @ R_invs[t] @ obs[t]
        else:
            Lambdas[t] = Lambda_1
            Thetas[t] = Theta_1


@njit(cache=True)
def forward_kalman_step(x_prev, A, Sigma_inv, Lambda_t, Theta_t):
    """Single forward Kalman sampling step (returns mean and precision).

    Parameters
    ----------
    x_prev : np.ndarray (D,)
        Previous state
    A : np.ndarray (D, D)
        Dynamics transition for current state
    Sigma_inv : np.ndarray (D, D)
        Inverse dynamics covariance
    Lambda_t : np.ndarray (D, D)
        Information matrix at time t
    Theta_t : np.ndarray (D,)
        Information vector at time t

    Returns
    -------
    mu : np.ndarray (D,)
        Posterior mean
    S : np.ndarray (D, D)
        Posterior covariance
    """
    # Posterior precision
    P = Sigma_inv + Lambda_t
    S = np.linalg.inv(P)

    # Posterior mean
    mu = S @ (Sigma_inv @ A @ x_prev + Theta_t)

    return mu, S


# =============================================================================
# TRANSITION/DURATION SAMPLING HELPERS
# =============================================================================

@njit(cache=True, parallel=True)
def compute_kappa_matrix(shifted_Zs, upper_limit):
    """Compute kappa matrix for stick-breaking transition sampling.

    Parameters
    ----------
    shifted_Zs : np.ndarray (N,)
        Shifted state indices
    upper_limit : int
        Number of stick-breaking levels

    Returns
    -------
    np.ndarray (N, upper_limit)
        Kappa matrix
    """
    N = shifted_Zs.shape[0]
    kappa = np.empty((N, upper_limit))

    for i in prange(N):
        z = shifted_Zs[i]
        for j in range(upper_limit):
            eq = 1.0 if z == j else 0.0
            ge = 1.0 if z >= j else 0.0
            kappa[i, j] = eq - 0.5 * ge

    return kappa


@njit(cache=True)
def outer_product_sum(X, weights):
    """Compute weighted sum of outer products: sum_i w_i * x_i @ x_i.T

    Parameters
    ----------
    X : np.ndarray (N, D)
        Data matrix
    weights : np.ndarray (N,)
        Weights for each row

    Returns
    -------
    np.ndarray (D, D)
        Weighted outer product sum
    """
    N, D = X.shape
    result = np.zeros((D, D))

    for n in range(N):
        w = weights[n]
        for i in range(D):
            for j in range(D):
                result[i, j] += w * X[n, i] * X[n, j]

    return result


# =============================================================================
# LOG-LIKELIHOOD COMPUTATION
# =============================================================================

@njit(cache=True, parallel=True)
def batch_gaussian_loglik(X, mus, Sigma_invs, log_dets):
    """Batch compute Gaussian log-likelihoods for multiple states.

    Parameters
    ----------
    X : np.ndarray (T, D)
        Observations
    mus : np.ndarray (K, T, D)
        Means for each state and time
    Sigma_invs : np.ndarray (K, D, D)
        Inverse covariances for each state
    log_dets : np.ndarray (K,)
        Log determinants of covariances

    Returns
    -------
    np.ndarray (K, T)
        Log-likelihoods for each state and time
    """
    K, T, D = mus.shape
    log_2_pi = 1.8378770664093453  # np.log(2 * np.pi)

    result = np.empty((K, T))

    for k in prange(K):
        for t in range(T):
            # Mahalanobis distance
            mahal = 0.0
            for i in range(D):
                diff_i = X[t, i] - mus[k, t, i]
                for j in range(D):
                    diff_j = X[t, j] - mus[k, t, j]
                    mahal += diff_i * Sigma_invs[k, i, j] * diff_j

            result[k, t] = -0.5 * (D * log_2_pi + log_dets[k] + mahal)

    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_numba_status():
    """Check Numba availability and configuration.

    Returns
    -------
    dict
        Status information about Numba configuration
    """
    if not NUMBA_AVAILABLE:
        return {
            "available": False,
            "message": "Numba not installed. Install with: pip install numba"
        }

    return {
        "available": True,
        "version": numba.__version__,
        "threading_layer": numba.config.THREADING_LAYER,
        "num_threads": numba.config.NUMBA_NUM_THREADS
    }


# Wrapper functions that check Numba availability and fall back to NumPy

def sigmoid(x):
    """Numerically stable sigmoid with optional Numba acceleration."""
    if NUMBA_AVAILABLE:
        return sigmoid_numba(np.asarray(x, dtype=np.float64))
    else:
        # Pure NumPy fallback
        result = np.zeros_like(x, dtype=np.float64)
        mask = x >= 0
        result[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        result[~mask] = np.exp(x[~mask]) / (1.0 + np.exp(x[~mask]))
        return result


# =============================================================================
# ADVANCED OPTIMIZED KERNELS (Added for performance optimization)
# =============================================================================

@njit(cache=True, parallel=True)
def batch_invert_psd(matrices):
    """Batch invert positive semi-definite matrices in parallel.

    Parameters
    ----------
    matrices : np.ndarray (N, D, D)
        Array of N matrices to invert

    Returns
    -------
    np.ndarray (N, D, D)
        Array of inverted matrices
    """
    N = matrices.shape[0]
    D = matrices.shape[1]
    result = np.empty((N, D, D))

    for i in prange(N):
        result[i] = np.linalg.inv(matrices[i])

    return result


@njit(cache=True)
def forward_kalman_sample_loop(T, As, Sigma_invs, Lambdas, Thetas, states,
                                x0, dyn_thetas, extended_kalman):
    """Full forward Kalman sampling loop with Cholesky-based MVN sampling.

    Parameters
    ----------
    T : int
        Number of timesteps
    As : np.ndarray (K, D, D)
        Dynamics matrices for each state
    Sigma_invs : np.ndarray (K, D, D)
        Pre-inverted dynamics covariances for each state
    Lambdas : np.ndarray (T+1, D, D)
        Information matrices from backward pass
    Thetas : np.ndarray (T+1, D)
        Information vectors from backward pass
    states : np.ndarray (T,)
        State sequence (integer)
    x0 : np.ndarray (D,)
        Initial state sample
    dyn_thetas : np.ndarray (T, D)
        Dynamics intercept terms (for affine dynamics)
    extended_kalman : bool
        Whether using extended Kalman filter

    Returns
    -------
    np.ndarray (T+1, D) if extended_kalman else (T, D)
        Sampled latent trajectory
    """
    D = x0.shape[0]
    T_ = T + 1 if extended_kalman else T
    res = np.zeros((T_, D))
    x = x0.copy()

    if extended_kalman:
        res[0] = x

    for t in range(T):
        t_ = t + 1 if extended_kalman else t
        state = int(states[t])
        Sigma_inv = Sigma_invs[state]
        A = As[state]

        if t_ > 0:
            # Posterior precision and covariance
            P = Sigma_inv + Lambdas[t_]
            S = np.linalg.inv(P)
            mu_new = S @ (Sigma_inv @ (A @ x - dyn_thetas[t]) + Thetas[t_])
        else:
            S = np.linalg.inv(Lambdas[t_])
            mu_new = S @ Thetas[t_]

        # Cholesky-based MVN sampling
        L = cholesky_lower(S)
        z = np.random.randn(D)
        x = cholesky_mvn_sample(mu_new, L, z)
        res[t_] = x

    return res


@njit(cache=True, parallel=True)
def batch_stick_breaking_logprobs(X_batch, R, K, loopy):
    """Batch stick-breaking log-probabilities for recurrent transitions.

    Parameters
    ----------
    X_batch : np.ndarray (T, D)
        Latent states for each timestep
    R : np.ndarray (K, D, upper_limit) or (1, D, upper_limit)
        Stick-breaking weights
    K : int
        Number of states
    loopy : bool
        Whether self-transitions are allowed

    Returns
    -------
    np.ndarray (T, K, K)
        Log transition probabilities for each timestep
    """
    T = X_batch.shape[0]
    D = X_batch.shape[1]
    upper_limit = K - 1 if loopy else K - 2
    nonswitching = R.shape[0] == 1

    result = np.empty((T, K, K))

    for t in prange(T):
        x = X_batch[t]

        for i in range(K):
            # Compute X_ = x @ R[i] or x @ R[0] if nonswitching
            r_idx = 0 if nonswitching else i
            X_ = np.zeros(upper_limit)
            for d in range(D):
                for u in range(upper_limit):
                    X_[u] += x[d] * R[r_idx, d, u]

            # Stick-breaking with log-sigmoid
            log_probs = np.zeros(K)
            cumsum_log_one_minus = 0.0

            for j in range(upper_limit):
                # Numerically stable log-sigmoid
                if X_[j] >= 0:
                    log_sig = -np.log(1.0 + np.exp(-X_[j]))
                    log_one_minus_sig = -X_[j] - np.log(1.0 + np.exp(-X_[j]))
                else:
                    log_sig = X_[j] - np.log(1.0 + np.exp(X_[j]))
                    log_one_minus_sig = -np.log(1.0 + np.exp(X_[j]))

                log_probs[j] = cumsum_log_one_minus + log_sig
                cumsum_log_one_minus += log_one_minus_sig

            # Last state gets remaining probability
            log_probs[upper_limit] = cumsum_log_one_minus

            # If not loopy, adjust indices to skip self-transition
            if not loopy:
                # Shift probabilities: j < i stays, j > i shifts
                temp = log_probs.copy()
                for j in range(K):
                    if j < i:
                        result[t, i, j] = temp[j]
                    elif j > i:
                        result[t, i, j] = temp[j - 1]
                    else:
                        result[t, i, j] = -1e10  # log(0) for self-transition
            else:
                for j in range(K):
                    result[t, i, j] = log_probs[j]

    return result


@njit(cache=True, parallel=True)
def batch_outer_product_sum(X, weights_batch, K):
    """Batch compute weighted outer product sums for each state.

    Parameters
    ----------
    X : np.ndarray (N, D)
        Data matrix
    weights_batch : np.ndarray (N, K)
        Weights for each observation and state
    K : int
        Number of states

    Returns
    -------
    np.ndarray (K, D, D)
        Weighted outer product sum for each state
    """
    N, D = X.shape
    result = np.zeros((K, D, D))

    for k in prange(K):
        for n in range(N):
            w = weights_batch[n, k]
            if w > 0:
                for i in range(D):
                    for j in range(D):
                        result[k, i, j] += w * X[n, i] * X[n, j]

    return result


@njit(cache=True)
def turn_matrix_psd_numba(S):
    """Numba-compatible function to turn a matrix positive semi-definite.

    Parameters
    ----------
    S : np.ndarray (D, D)
        Input matrix

    Returns
    -------
    np.ndarray (D, D)
        Positive semi-definite matrix
    """
    # Clip extreme values
    D = S.shape[0]
    for i in range(D):
        for j in range(D):
            if S[i, j] > 1.e6:
                S[i, j] = 1.e6
            elif S[i, j] < -1.e6:
                S[i, j] = -1.e6

    # Eigenvalue decomposition
    U, V = np.linalg.eig(S)

    # Reconstruct with absolute eigenvalues
    result = np.zeros((D, D), dtype=np.complex128)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                result[i, j] += V[i, k] * np.abs(U[k]) * V[j, k]

    return result.real.astype(np.float64)


@njit(cache=True)
def robust_cholesky_sample(mu, S, max_attempts=3):
    """Sample from MVN with robust Cholesky handling.

    Parameters
    ----------
    mu : np.ndarray (D,)
        Mean vector
    S : np.ndarray (D, D)
        Covariance matrix
    max_attempts : int
        Maximum number of PSD fix attempts

    Returns
    -------
    np.ndarray (D,)
        Sample from N(mu, S)
    """
    D = mu.shape[0]
    z = np.random.randn(D)

    # Try Cholesky directly
    L = cholesky_lower(S)

    # Check if Cholesky succeeded (all diagonal elements positive)
    valid = True
    for i in range(D):
        if L[i, i] <= 0:
            valid = False
            break

    if valid:
        return cholesky_mvn_sample(mu, L, z)

    # Fallback: Add small diagonal term
    S_fixed = S.copy()
    for attempt in range(max_attempts):
        eps = 1e-6 * (10 ** attempt)
        for i in range(D):
            S_fixed[i, i] = S[i, i] + eps

        L = cholesky_lower(S_fixed)
        valid = True
        for i in range(D):
            if L[i, i] <= 0:
                valid = False
                break

        if valid:
            return cholesky_mvn_sample(mu, L, z)

    # Last resort: use turn_matrix_psd_numba
    S_psd = turn_matrix_psd_numba(S)
    L = cholesky_lower(S_psd)
    return cholesky_mvn_sample(mu, L, z)
