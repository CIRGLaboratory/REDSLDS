import logging

import numpy as np
from pathos.helpers import mp
from pathos.multiprocessing import ProcessingPool

from .decisionlist import DecisionList

# Try to import Numba-accelerated functions
try:
    from .numba_kernels import NUMBA_AVAILABLE, sigmoid_numba
except ImportError:
    NUMBA_AVAILABLE = False

def types(**_params_):
    def check_types(_func_, _params_=_params_):
        def modified(*args, **kw):
            arg_names = _func_.func_code.co_varnames
            kw.update(zip(arg_names, args))
            for name, type in _params_.items():
                param = kw[name]
                assert param is None or isinstance(param, type), \
                    "Parameter '%s' should be type '%s', and is currently '%s'" \
                    % (name, type.__name__, param.__class__)
            return _func_(**kw)

        return modified

    return check_types


def isprob(x):
    return (x.sum() > 0.9) and (x.sum() < 1.1)


def bayesian_linear_regression_posterior(params, states, X, Y, L):
    '''

    Parameters
    ----------
    params
    states
    X
    Y
    L

    Returns
    -------

    '''
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(states, np.ndarray)
    for l in range(L):
        x = X[states == l]
        nu_0 = params["nu_0"][l]
        V_0 = params["V_0"][l]
        B_0 = params["B_0"][l]
        Lambda_0 = params["Lambda_0"][l]
        if x.shape[0] < 1:
            params["B_n"][l] = B_0
            params["V_n"][l] = V_0
            params["nu_n"][l] = nu_0
            params["Lambda_n"][l] = Lambda_0
            continue
        y = Y[states == l]
        B_n = invert(x.T @ x + Lambda_0) @ (x.T @ y + Lambda_0 @ B_0)
        V_n = V_0 + (y - x @ B_n).T @ (y - x @ B_n) + (B_n - B_0).T @ Lambda_0 @ (B_n - B_0)
        nu_n = nu_0 + y.shape[0]
        Lambda_n = x.T @ x + Lambda_0
        params["B_n"][l] = B_n
        params["V_n"][l] = V_n
        params["nu_n"][l] = nu_n
        params["Lambda_n"][l] = Lambda_n
    return params


def pool_size(num_workers, n_tasks):
    return max(1, min(int(num_workers), int(n_tasks)))


log = logging.getLogger("edhmm")


class WorkerTimeout(RuntimeError):
    """A pool call did not return within TimedPool.TIMEOUT_SECONDS twice: a worker deadlocks or died."""


class TimedPool(ProcessingPool):
    """pathos pool whose ``map`` does not wait forever on a deadlocked forked worker.

    A call that does not return within TIMEOUT_SECONDS gets its workers
    terminated and the cached pool dropped, and runs once more with fresh
    workers; if that times out too, ``WorkerTimeout`` is raised.
    """

    TIMEOUT_SECONDS = 3600.     # the slowest pool call of the 2026-08-27 matrix took 93 s (bee_seq_data_full)

    def map(self, f, *args, **kwds):
        for retry in (False, True):
            try:
                return self.amap(f, *args, **kwds).get(self.TIMEOUT_SECONDS)
            except mp.TimeoutError:
                self.terminate()
                self.clear()
                if retry:
                    raise WorkerTimeout(f"pool call did not return within {self.TIMEOUT_SECONDS:.0f} s, twice") from None
                log.warning("pool call did not return within %.0f s; retrying with fresh workers", self.TIMEOUT_SECONDS)


def worker_pool(num_workers, n_tasks):
    """pathos pool for ``n_tasks`` parallel tasks, never larger than the task count.

    pathos forks the workers when the pool object is built, keeps every pool
    (keyed by worker count) alive for the whole run and its ``__exit__`` is a
    no-op, so each worker costs its copy-on-write share of the parent plus its
    own heap high-water mark for as long as the process lives, idle or not.
    The sampler therefore holds two resident pools: one sized by the number of
    sequences, one by K.
    """
    return TimedPool(pool_size(num_workers, n_tasks))


def invert_diagonal(diag):
    """Inverse of ``np.diag(diag)`` as a vector (Polya-Gamma and gamma draws are strictly positive)."""
    return 1. / np.asarray(diag, dtype=float)


def invert(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.solve(A, np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            try:
                return np.linalg.lstsq(A, np.eye(A.shape[0]))[0]
            except np.linalg.LinAlgError as e:
                print(f"Error while inverting matrix: {A}")
                raise e


def invert_3d(A):
    """Compute the inverse of matrices in an array of shape (N,N,M)"""
    return np.linalg.inv(A.transpose(2, 0, 1)).transpose(1, 2, 0)


def turn_matrix_positive_semidefinite(S):
    S = np.minimum(S, 1.e6)
    S = np.maximum(S, -1.e6)
    U, V = np.linalg.eig(S)
    return V @ np.diag(np.abs(U)) @ V.T

def make_positive_definite(S, rel_floor=1e-8, abs_floor=1e-10):
    """Symmetric positive-definite version of S: eigenvalues floored to rel_floor * max |eigenvalue|.

    The *_turn_matrix_positive_semidefinite repairs only guarantee semi-definite
    (and perturb the eigenvectors), which scipy's Cholesky still rejects for a
    near-singular posterior covariance; this is the deterministic last resort.
    """
    S = np.clip(np.asarray(S, dtype=float), -1.e6, 1.e6)
    S = (S + S.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    floor = max(rel_floor * np.max(np.abs(eigenvalues)), abs_floor)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def nearest_psd(matrix):
    # Symmetrize the matrix
    matrix = (matrix + matrix.T) / 2

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Set negative eigenvalues to zero
    eigenvalues[eigenvalues < 0] = 0

    # Reconstruct the matrix
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return psd_matrix

def aggresively_turn_matrix_positive_semidefinite(S):
    S = np.minimum(S, 1.e6)
    S = np.maximum(S, -1.e6)
    U, V = np.linalg.eig(S)
    V = V + np.random.normal(scale=0.001, size=V.shape)
    return V @ np.diag(np.abs(U)) @ V.T

def super_aggresively_turn_matrix_positive_semidefinite(S):
    S = np.minimum(S, 1.e6)
    S = np.maximum(S, -1.e6)
    S = S + np.eye(S.shape[0]) * 0.01
    U, V = np.linalg.eig(S)
    V = V + np.random.normal(scale=0.001, size=V.shape)
    return V @ np.diag(np.abs(U)) @ V.T

def sigmoid(x):
    """Numerically stable sigmoid function - uses Numba when available."""
    if NUMBA_AVAILABLE:
        return sigmoid_numba(np.asarray(x, dtype=np.float64))
    # Pure NumPy fallback
    res = np.zeros_like(x, dtype=np.float64)
    mask = x >= 0
    res[mask] = 1 / (1 + np.exp(-x[mask]))
    res[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
    return res

def relabel_by_permutation(l, perm):
    out = np.empty_like(l)
    good = ~np.isnan(l)
    out[good] = perm[l[good].astype('int32')]
    if np.isnan(l).any():
        out[~good] = np.nan
    return out

def fit_decision_list(Z, X, K, affine=False, inputs=None):
    print("Fitting Decision List")
    D_latent = X[0].shape[1]
    dlist = DecisionList(K, D_latent,lr_params=dict(penalty="l2",
                                          fit_intercept=affine,
                                          C=100.) )
    X_concat = np.concatenate([x[:-1] for x in X])
    if inputs is not None:
        inputs = np.array(inputs)
        X_concat = np.concatenate([X_concat, inputs[:-1]], axis=1)
    Z_concat = np.concatenate([z[1:] for z in Z])
    dlist.fit(X_concat, Z_concat)
    D_ = D_latent if not affine else D_latent+1
    R_res = np.zeros((D_, K-1))
    R_res[:D_latent] = dlist.weights.T[:D_latent]
    if affine:
        R_res[D_latent] = dlist.biases
    if inputs is not None:
        R_res[D_latent+1:] = dlist.weights.T[D_latent:]
    Z_perm = [relabel_by_permutation(z, np.argsort(dlist.permutation)) for z in Z]
    return Z_perm, R_res
