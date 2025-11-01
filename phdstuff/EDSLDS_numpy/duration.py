from collections import defaultdict as dd
import logging
import os.path
import pickle
from typing import List

from icecream import ic
import numpy as np
from .utils import worker_pool as Pool
from polyagamma import random_polyagamma
from scipy.stats import dirichlet
from scipy.stats import gamma
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import poisson

from phdstuff.utils import log_wandb_scalar_or_array

from .utils import aggresively_turn_matrix_positive_semidefinite
from .utils import invert
from .utils import sigmoid
from .utils import super_aggresively_turn_matrix_positive_semidefinite
from .utils import turn_matrix_positive_semidefinite

log = logging.getLogger('duration')
FPATH = "../Data/ftable.npy"


try:
    F = np.load(FPATH)
except Exception:
    F = np.zeros((1, 1))


def categorical(alpha):
    return multinomial.rvs(1, alpha / np.sum(alpha)).nonzero()[0][0]


class NonRecurrentDuration:
    def __init__(self, D):
        self.D = D
        self.affine = False

    def sample_Omegas(self, Zs, Xs, Ds, inputs=None):
        return np.zeros_like(Zs).reshape((-1, 1))

    def get_message(self, Zs, Ds, Omegas, epsilons, switches, inputs = None):
        # if len(switches) == len(Zs):
        #     switches = switches[:-1]
        N = len(Zs) + 1
        return np.zeros((N, self.D)), np.zeros((N, self.D, self.D))

    def init_pg(self, size):
        return random_polyagamma(size=size)

    def rotate(self, rotation):
        pass


class newPoisson(NonRecurrentDuration):
    def __init__(self, mu, alpha, beta, D, support_step=1):
        assert isinstance(mu, np.ndarray)
        super().__init__(D)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.states = np.arange(len(mu))
        self.support_step = support_step

    def loglikelihood(self, state, k):
        assert state in self.states
        return (k * np.log(self.mu[state])) - np.log(np.math.factorial(k)) - self.mu[state]

    def sample_d(self, state):
        # return int(tfd.Poisson(self.mu[state]).sample())
        return int(poisson.rvs(self.mu[state]))

    def sample_mu(self, Zs):
        k = dd(list)

        for Z in Zs:
            now = Z[0]
            for s in Z:
                if now == s:
                    try:
                        k[now][-1] += 1
                    except IndexError:
                        # initial condition
                        k[now] = [1]
                else:
                    now = s
                    k[now].append(1)

        for i in self.states:
            log.debug("state: %s" % i)
            log.debug("observations: %s" % k[i])

        out = []
        for i in self.states:
            alpha = self.alpha[i] + sum(k[i])
            beta = self.beta[i] + len(k[i])
            log.debug('drawing mu from a gamma with alpha=%s and beta=%s' % (alpha, beta))
            # out.append(tfd.Gamma(alpha, beta).sample())
            out.append(gamma.rvs(alpha, scale=1. / beta))
            log.debug('sampled rate parameter for state %s: %s' % (i, out[-1]))
        return out

    def update(self, Z):
        self.mu = self.sample_mu(Z)

    def support(self, state, threshold=0.00001):
        log.info('finding support for state %s' % state)
        # walk left
        d, dl = int(self.mu[state]), 1
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            lnew = self.loglikelihood(state, d)  # TODO: Tu jest coś bardzo skopane
            dl = abs(lnew - lold)
            lold = lnew
            d -= self.support_step
        left = max(1, d)
        # walk right
        d, dl = int(self.mu[state]), 1
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            d += self.support_step
            lnew = np.exp(self.loglikelihood(state, d))
            dl = abs(lnew - lold)
            lold = lnew
        right = max(1, d)
        log.debug('support for state %s: %s to %s' % (state, int(left), int(right)))
        return int(left), int(right)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        import wandb
        wandb.log({
            "duration_mu": self.mu
        }, step=step, commit=commit, sync=sync)


class DummyDuration(NonRecurrentDuration):

    def __init__(self, K=None, D=None, recurrent=False):
        super().__init__(D)
        self.K = K
        self.recurrent = recurrent

    def loglikelihood(self, state, k, *args, **kwargs):
        if k == 1:
            return 0.
        else:
            return np.log(0.0000000000000000000000000000000001)

    def sample_d(self, *args, **kwargs):
        return 1

    def max_support(self):
        return 1, 1

    def update(self, *args, **kwargs):
        pass

    def support(self, *args, **kwargs):
        return 1, 2

    def log_to_wandb(self, step=None, commit=None, sync=None):
        pass

    def dump(self, path, name):
        pass

    @staticmethod
    def load(path, name):
        return DummyDuration()


class CategoricalDuration(NonRecurrentDuration):
    def __init__(self, K, d_max, D, alpha=None, log_transition_matrix=None):
        super().__init__(D)
        self.d_max = d_max
        self.K = K
        if log_transition_matrix is None:
            log_transition_matrix = np.log(np.ones((K, d_max)) / d_max)
        self.log_transition_matrix = log_transition_matrix
        self.states = range(K)
        if alpha is None:
            alpha = np.ones((K, d_max)) / d_max
        self.alpha = alpha

    def loglikelihood(self, state, k):
        return self.log_transition_matrix[state, k - 1]

    def sample_d(self, state):
        return int(categorical(np.exp(self.log_transition_matrix[state]))) + 1

    def sample_log_trans(self, Zs):

        n = dict([(i, dict([(j, 0) for j in range(self.d_max)])) for i in self.states])

        for Z in Zs:
            x, now_dur = Z[0]  # TODO: Sprawdzić to
            now = x
            for (x, d) in Z:
                if now_dur <= d:
                    n[now][d - 1] += 1
                    now = x
                now_dur = d

        A = np.zeros((self.K, self.d_max))

        for i in self.states:
            # A[i] = (dirichlet.rvs(self.alpha[i] + np.array(list(n[i].values()))))
            A[i] = (dirichlet.rvs((self.alpha[i] + np.array(list(n[i].values())))))

        log.debug('sampled A:\n%s' % A)

        return np.log(A)

    def update(self, Z, num_workers=5, z_tuples=True):
        self.log_transition_matrix = self.sample_log_trans(Z)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(self.log_transition_matrix, "duration_matrix", step=step, commit=commit, sync=sync)

    def support(self, *args, **kwargs):
        return 1, self.d_max

    def max_support(self):
        return 1, self.d_max

    def dump(self, path, name):
        dump_dic = {
            "alpha": self.alpha,
            "d_max": self.d_max,
            "K": self.K,
            "log_transition_matrix": self.log_transition_matrix
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_cat_dur.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_cat_dur.pickle"), "rb") as ifile:
            init_dic = pickle.load(ifile)
        return CategoricalDuration(**init_dic)


class Poisson(NonRecurrentDuration):
    def __init__(self, mu, alpha, beta, D, support_step=1):
        super().__init__(D)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.states = range(len(mu))
        self.support_step = support_step

    def loglikelihood(self, state, k):
        assert state in self.states
        return (k * np.log(self.mu[state])) - sum([np.log(ki + 1) for ki in range(k)]) - self.mu[state]

    def sample_d(self, state):
        return int(poisson.rvs(self.mu[state]))

    def sample_mu(self, Zs):

        k = dict([(i, []) for i in self.states])

        for Z in Zs:
            X = [z[0] for z in Z]
            now = X[0]
            for s in X:
                if now == s:
                    try:
                        k[now][-1] += 1
                    except IndexError:
                        # initial condition
                        k[now] = [1]
                else:
                    now = s
                    k[now].append(1)

        for i in self.states:
            log.debug("state: %s" % i)
            log.debug("observations: %s" % k[i])

        out = []
        for i in self.states:
            alpha = self.alpha[i] + sum(k[i])
            beta = self.beta[i] + len(k[i])
            log.debug('drawing mu from a gamma with alpha=%s and beta=%s' % (alpha, beta))
            out.append(gamma.rvs(alpha, scale=1. / beta))
            log.debug('sampled rate parameter for state %s: %s' % (i, out[-1]))
        return out

    def update(self, Z):
        self.mu = self.sample_mu(Z)

    def support(self, state, threshold=0.00001):
        log.info('finding support for state %s' % state)
        # walk left
        d, dl = int(self.mu[state]), 1
        lold = self.loglikelihood(state, d)
        while (dl > threshold) and (d > 0):
            lnew = self.loglikelihood(state, d)  # TODO: Tu jest coś bardzo skopane
            dl = abs(lnew - lold)
            lold = lnew
            d -= self.support_step
        left = max(1, d)
        # walk right
        d, dl = int(self.mu[state]), 1
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            d += self.support_step
            lnew = np.exp(self.loglikelihood(state, d))
            dl = abs(lnew - lold)
            lold = lnew
        right = max(1, d)
        log.debug('support for state %s: %s to %s' % (state, int(left), int(right)))
        return int(left), int(right)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        import wandb

        wandb.log({
            "duration_mu": self.mu
        }, step=step, commit=commit, sync=sync)


class TestPoisson(NonRecurrentDuration):
    def __init__(self, mu=None, alpha=None, beta=None, D=None, support_step=1, K=None):
        super().__init__(D)
        if mu is None:
            mu = np.ones(K)
        else:
            K = len(mu)
        self.K = K
        if alpha is None:
            alpha = np.ones(K)
        if beta is None:
            beta = np.ones(K)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.states = range(len(mu))
        self.support_step = support_step
        self.D_max = 200

    def dump(self, path, name):
        dump_dic = {
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "support_step": self.support_step
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_dur.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_dur.pickle"), "rb") as ifile:
            init_dic = pickle.load(ifile)
        return TestPoisson(**init_dic)

    def loglikelihood(self, state, k):
        assert state in self.states
        k = k - 1
        return poisson.logpmf(k, self.mu[state])

    def sample_d(self, state):
        return int(poisson.rvs(self.mu[state])) + 1

    def _sample_mu(self, k):
        def _res(i):
            alpha = self.alpha[i] + sum(k[i])
            beta = self.beta[i] + len(k[i])
            log.debug('drawing mu from a gamma with alpha=%s and beta=%s' % (alpha, beta))
            res = gamma.rvs(alpha, scale=1. / beta)
            log.debug('sampled rate parameter for state %s: %s' % (i, res))
            return res

        return _res

    def max_support(self):
        return 1, self.D_max

    def sample_mu(self, Zs, num_workers=5, z_tuples=True):
        k = dict([(i, []) for i in self.states])

        for Z in Zs:
            if z_tuples:
                X = [z[0] for z in Z]
            else:
                X = Z
            now = -1
            for s in X:
                if now == s:
                    try:
                        k[now][-1] += 1
                    except IndexError:
                        # initial condition
                        k[now] = [0]
                else:
                    now = s
                    k[now].append(0)

        # out = []
        # for i in self.states:
        #     alpha = self.alpha[i] + sum(k[i])
        #     beta = self.beta[i] + len(k[i])
        #     log.debug('drawing mu from a gamma with alpha=%s and beta=%s' % (alpha, beta))
        #     out.append(gamma.rvs(alpha, scale = 1./beta))
        #     log.debug('sampled rate parameter for state %s: %s' % (i, out[-1]))
        #     if out[-1] > 1000:
        #         print(alpha)  # 1
        #         print(beta)  # 0.00001
        with Pool(num_workers, len(self.states)) as pool:
            out = pool.map(self._sample_mu(k), self.states)
        return out

    def update(self, Z, num_workers=5, z_tuples=True):
        self.mu = self.sample_mu(Z, num_workers, z_tuples)

    def support(self, state, threshold=0.00001, cache=None):
        log.info('finding support for state %s' % state)
        # walk left
        d, dl = int(self.mu[state]) + 1, 1
        if (cache is not None) and (d in cache):
            return cache[d]
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            lnew = self.loglikelihood(state, d)  # TODO: Tu jest coś bardzo skopane
            dl = abs(lnew - lold)
            lold = lnew
            d -= self.support_step
        left = max(1, d)
        # walk right
        d, dl = int(self.mu[state]) + 1, 1
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            d += self.support_step
            lnew = np.exp(self.loglikelihood(state, d))
            dl = abs(lnew - lold)
            lold = lnew
        right = max(1, d)
        log.debug('support for state %s: %s to %s' % (state, int(left), int(right)))
        if cache is not None:
            cache[int(self.mu[state]) + 1] = (int(left), int(right))
        return int(left), int(right)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(
            self.mu, "duration_mu",
            step=step, commit=commit, sync=sync)


class LoopyPoisson(NonRecurrentDuration):
    def __init__(self, mu=None, alpha=None, beta=None, D=None, support_step=1, K=None):
        super().__init__(D)
        if mu is None:
            mu = np.ones(K)
        else:
            K = len(mu)
        if alpha is None:
            alpha = np.ones(K)
        if beta is None:
            beta = np.ones(K)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.states = range(len(mu))
        self.support_step = support_step
        self.D_max = 200

    def max_support(self):
        return 1, self.D_max

    def dump(self, path, name):
        dump_dic = {
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "support_step": self.support_step
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_dur.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_dur.pickle"), "rb") as ifile:
            init_dic = pickle.load(ifile)
        return TestPoisson(**init_dic)

    def loglikelihood(self, state, k):
        assert state in self.states
        k = k - 1
        return poisson.logpmf(k, self.mu[state])

    def sample_d(self, state):
        return int(poisson.rvs(self.mu[state])) + 1

    def _sample_mu(self, k):
        def _res(i):
            alpha = self.alpha[i] + sum(k[i])
            beta = self.beta[i] + len(k[i])
            log.debug('drawing mu from a gamma with alpha=%s and beta=%s' % (alpha, beta))
            res = gamma.rvs(alpha, scale=1. / beta)
            log.debug('sampled rate parameter for state %s: %s' % (i, res))
            return res

        return _res

    def sample_mu(self, Zs, num_workers=5, z_tuples=True):
        k = dict([(i, []) for i in self.states])
        assert z_tuples
        for Z in Zs:
            now, now_dur = -1, -1
            for s, d in Z:
                if now_dur == d + 1:
                    try:
                        k[now][-1] += 1
                    except IndexError:
                        # initial condition
                        k[now] = [0]
                else:
                    now = s
                    k[now].append(0)
                now_dur = d

        # out = []
        # for i in self.states:
        #     alpha = self.alpha[i] + sum(k[i])
        #     beta = self.beta[i] + len(k[i])
        #     log.debug('drawing mu from a gamma with alpha=%s and beta=%s' % (alpha, beta))
        #     out.append(gamma.rvs(alpha, scale = 1./beta))
        #     log.debug('sampled rate parameter for state %s: %s' % (i, out[-1]))
        #     if out[-1] > 1000:
        #         print(alpha)  # 1
        #         print(beta)  # 0.00001
        with Pool(num_workers, len(self.states)) as pool:
            out = pool.map(self._sample_mu(k), self.states)
        return out

    def update(self, Z, num_workers=5, z_tuples=True):
        self.mu = self.sample_mu(Z, num_workers, z_tuples)

    def support(self, state, threshold=0.00001, cache=None):
        log.info('finding support for state %s' % state)
        # walk left
        d, dl = int(self.mu[state]) + 1, 1
        if cache is not None and d in cache:
            return cache[d]
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            lnew = self.loglikelihood(state, d)  # TODO: Tu jest coś bardzo skopane
            dl = abs(lnew - lold)
            lold = lnew
            d -= self.support_step
        left = max(1, d)
        # walk right
        d, dl = int(self.mu[state]) + 1, 1
        lold = self.loglikelihood(state, d)
        while dl > threshold:
            d += self.support_step
            lnew = np.exp(self.loglikelihood(state, d))
            dl = abs(lnew - lold)
            lold = lnew
        right = max(1, d)
        log.debug('support for state %s: %s to %s' % (state, int(left), int(right)))
        if cache is not None:
            cache[int(self.mu[state]) + 1] = (int(left), int(right))
        return int(left), int(right)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(
            self.mu, "duration_mu",
            step=step, commit=commit, sync=sync)


class RecurrentDuration:
    def __init__(self, K: int, D: int, D_max: int, R=None, R_mu=None, R_sigma=None, alpha=None, eps=4.,
                 affine=False, nonswitching=True, input_dim=None) -> None:
        if alpha is None:
            alpha = [np.array([1.0 / (K) for i in range(K)]) for j in range(K)]
        self.affine = affine
        self.alpha = alpha
        self.K = K
        self.D = D
        self.states = range(K)
        self.D_max = D_max
        self.nonswitching = nonswitching
        self.input_dim = input_dim
        D_ = D if not affine else D + 1
        if input_dim is not None:
            D_ = D_ + input_dim
        if R is None:
            if nonswitching:
                R = np.ones((1, D_, D_max - 1))
            else:
                R = np.stack([np.ones((D_, D_max - 1)) for _ in self.states], axis=0)
        self.R = R
        if R_mu is None:
            if nonswitching:
                R_mu = np.zeros((1, D_, D_max - 1))
            else:
                R_mu = np.stack([np.zeros((D_, D_max - 1)) for _ in self.states], axis=0)
        self.R_mu = R_mu
        if R_sigma is None:
            if nonswitching:
                R_sigma = np.tile(np.eye(D_), (D_max - 1, 1, 1)).transpose((1, 2, 0)) * eps
                R_sigma = R_sigma[np.newaxis]
            else:
                R_sigma = np.stack(
                    [np.tile(np.eye(D_), (D_max - 1, 1, 1)).transpose((1, 2, 0)) * eps for _ in self.states],
                    axis=0)
        self.R_sigma = R_sigma

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
            "alpha": self.alpha,
            "R": self.R,
            "D": self.D,
            "R_mu": self.R_mu,
            "D_max": self.D_max,
            "R_sigma": self.R_sigma,
            "affine": self.affine,
            "nonswitching": self.nonswitching,
            "input_dim": self.input_dim
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_rec_tran.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_tran.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        return RecurrentDuration(**kwargs)

    def max_support(self):
        return 1, self.D_max

    def get_log_transition(self, i, X, omega=None, input=None):
        # Validate state index
        assert i in range(self.K), f"State index i={i} out of bounds for K={self.K}"

        if self.nonswitching:
            R = self.R[0]
        else:
            R = self.R[i]
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if self.affine and (X.shape[1] == self.D):
            X = np.concatenate([X, np.ones(X.shape[0])[:, np.newaxis]], axis=1)
        if input is not None:
            X = np.concatenate([X, input], axis=1)
        # if (self.K == 2):
        #     res = np.zeros((X.shape[0], self.D_max))
        #     res[:, i] = np.log(0.)
        #     return res.squeeze()
        if omega is None:
            if len(R.shape) == 3:
                X_ = (X[:, np.newaxis, :] @ R).squeeze()
            else:
                X_ = X @ R
            upper_limit = self.D_max - 1
            p = np.zeros((X_.shape[0], upper_limit + 1))
            mask = (np.eye(upper_limit) * 2 - np.tril(np.ones(upper_limit)))
            p_ = sigmoid(mask[np.newaxis, :, :] * X_[:, np.newaxis, :])
            # Numerical safeguard: clamp sigmoid output to avoid log(0) = -inf
            p_ = np.clip(p_, 1e-10, 1.0 - 1e-10)
            p_ = np.log(p_)
            p_ = np.tril(p_)
            p_ = np.sum(p_, axis=-1)
            p[:, :-1] = p_
            # Numerical safeguard: clamp sigmoid output for final duration
            sig_neg_X = sigmoid(-X_)
            sig_neg_X = np.clip(sig_neg_X, 1e-10, 1.0 - 1e-10)
            p[:, -1] = np.sum(np.log(sig_neg_X), axis=-1)
            assert p.shape[0] == X.shape[0]
        else:
            if len(omega.shape) == 2:
                omega = omega[np.newaxis, :, :]
            # Validate omega shape after potential reshape
            assert len(omega.shape) == 3, f"Omega must be 3D after reshape, got shape {omega.shape}"
            assert omega.shape[2] == self.D_max - 1, \
                f"Omega duration dimension wrong: expected {self.D_max - 1}, got {omega.shape[2]}"
            if len(R.shape) == 3:
                X_ = (X[:, np.newaxis, :] @ R).squeeze()
            else:
                X_ = X @ R
            upper_limit = self.D_max - 1
            states0 = np.arange(upper_limit)
            states1 = np.arange(upper_limit + 1)
            # For nonswitching, all states share the same omega (use index 0)
            # For switching, use state-specific omega (use index i)
            state_idx = 0 if self.nonswitching else i
            assert omega.shape[1] > state_idx, f"Omega state dimension {omega.shape[1]} <= state_idx {state_idx} (i={i}, nonswitching={self.nonswitching})"
            Omega_ = omega[:, state_idx]
            assert Omega_.shape == X_.shape, (Omega_.shape, X_.shape)
            Kappa = (states1[:, np.newaxis] == states0[np.newaxis]).astype(float) - 0.5 * (
                    states1[:, np.newaxis] >= states0[np.newaxis]).astype(float)
            potentials = Kappa[np.newaxis] * X_[:, np.newaxis, :] - 0.5 * (
                    (states1[np.newaxis, :, np.newaxis] >= states0[np.newaxis, np.newaxis]) * (Omega_ * X_ * X_)[:,
                                                                                              np.newaxis, :])
            potentials = potentials - np.max(potentials)
            potentials = np.exp(np.sum(potentials, axis=-1))
            if np.all(potentials == 0.):
                potentials = np.ones_like(potentials)
            p = (np.log(potentials) - np.log(np.sum(potentials, axis=1, keepdims=True))).reshape((X_.shape[0], -1))

        # Validate probabilities sum to 1
        assert np.all(np.isclose(np.sum(np.exp(p.squeeze()), axis=-1),
                                 np.ones_like(np.sum(np.exp(p.squeeze()), axis=-1)))), np.exp(p.squeeze())
        return p.squeeze()

    def loglikelihood(self, i, d, x, omega=None, input=None):
        assert i in self.states
        assert d <= self.D_max
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if self.affine and x.shape[1] == self.D:
            x = np.concatenate([x, np.ones(x.shape[0])[:, np.newaxis]], axis=1)
        if input is not None:
            x = np.concatenate([x, input], axis=1)
        return self.get_log_transition(i, x)[d - 1]

    def rotate(self, rotation):
        self.R[:, :self.D] = rotation[np.newaxis, :, :] @ self.R[:, :self.D]
        if self.affine:
            self.R[:, -1] = (rotation[np.newaxis, :, :] @ self.R[:, -1]).squeeze()

    def sample_d(self, i, X, omega=None, input=None):
        p = self.get_log_transition(i, X, input=input)
        p = np.exp(p)
        if len(p.shape) > 1:
            p = p.reshape(-1)
        try:
            return categorical(p) + 1
        except Exception:
            print(p)
            p = p - np.finfo(np.float32).epsneg
            p = np.absolute(p)
            return categorical(p) + 1

    def _sample_r(self, Zs, Ds, Xs, Omegas, input=None):
        def _res(k):
            if self.nonswitching:
                indices = np.ones_like(Zs).astype(bool)
            else:
                indices = Zs == k
            upper_limit = self.D_max - 1
            # indices[1:] = indices[1:] & (Ds[1:] >= Ds[:-1])
            Kappas = np.zeros((np.sum(indices), upper_limit))
            Xs_ = Xs[indices]
            Omegas_ = Omegas[indices]
            assert not np.any(np.isnan(Xs_)), f"Z: {Xs_}"
            assert not np.any(np.isnan(Omegas_)), f"Z: {Omegas_}"
            if len(Xs_) == 0:
                return self.R[k]
            R = self.R[k].copy()
            Ds_ = Ds[indices] - 1
            for j in range(upper_limit):
                # Match RecurrentTransition pattern: explicit float conversion for consistency
                Kappas[:, j] = (Ds_ == j).astype(float) - 0.5 * (Ds_ >= j).astype(float)
                Omegas_j = Omegas_[:, j] * (Ds_ >= j).astype(float)
                mu0 = self.R_mu[k][:, j]
                precision0 = invert(self.R_sigma[k][:, :, j])
                theta0 = precision0 @ mu0
                Xs_prime = np.sqrt(Omegas_j[:, np.newaxis]) * Xs_
                D_ = self.D if not self.affine else self.D + 1
                D_ = D_ if input is None else D_ + self.input_dim
                precision = precision0 + np.einsum('ni,nj->ij', Xs_prime, Xs_prime)
                sigma = invert(precision)
                theta = theta0 + np.sum(Kappas[:, j].reshape((-1, 1)) * Xs_, axis=0)
                mu = sigma @ theta
                try:
                    try:
                        R[:, j] = multivariate_normal.rvs(mean=mu, cov=sigma)
                    except Exception:
                        sigma = turn_matrix_positive_semidefinite(sigma)
                        mu = sigma @ theta
                        try:
                            R[:, j] = multivariate_normal.rvs(mean=mu, cov=sigma)
                        except Exception:
                            sigma = aggresively_turn_matrix_positive_semidefinite(sigma)
                            mu = sigma @ theta
                            try:
                                R[:, j] = multivariate_normal.rvs(mean=mu, cov=sigma)
                            except Exception:
                                sigma = super_aggresively_turn_matrix_positive_semidefinite(sigma)
                                mu = sigma @ theta
                                R[:, j] = multivariate_normal.rvs(mean=mu, cov=sigma)
                except Exception as e:
                    ic(sigma)
                    ic(mu)
                    ic(Xs_prime)
                    ic(precision)
                    ic(theta)
                    raise e
            return R

        return _res

    def sample_params(self, Zs: List[np.ndarray], Xs: List[np.ndarray], Ds: List[np.ndarray], Omegas: List[np.ndarray],
                      switches: List[np.ndarray], num_workers=32, input=None):
        if self.nonswitching or any(Omega_.shape[1] == 1 for Omega_ in Omegas):
            try:
                Omegas = np.concatenate([Omega_[:, 0][switches_] for Omega_, switches_ in zip(Omegas, switches)])
            except Exception:
                Omegas = np.concatenate([Omega_[:-1, 0][switches_[1:]] for Omega_, switches_ in zip(Omegas, switches)])
        else:
            try:
                Omegas = np.concatenate(
                    [np.take_along_axis(Omega_,
                                      Zs_[:, np.newaxis, np.newaxis],
                                      axis=1).squeeze()[switches_] for
                     Omega_, Zs_, switches_ in zip(Omegas, Zs, switches)])
            except Exception:
                Omegas = np.concatenate([np.take_along_axis(Omega_[:-1],
                                                          Zs_[:-1, np.newaxis, np.newaxis],
                                                          axis=1).squeeze()[switches_[1:]] for
                                         Omega_, Zs_, switches_ in zip(Omegas, Zs, switches)])
        Z = np.concatenate([Zs_[switches_] for Zs_, switches_ in zip(Zs, switches)])  # TODO: sprawdzic indeksy
        X = np.concatenate([Xs_[:-1][switches_] for Xs_, switches_ in zip(Xs, switches)])
        D = np.concatenate([Ds_[switches_] for Ds_, switches_ in zip(Ds, switches)])
        input_data = None
        if input is not None:
            input_data = np.concatenate([inp_[switches_] for inp_, switches_ in zip(input, switches)])
        if self.affine and (X.shape[1] == self.D):
            X = np.concatenate([X, np.ones(X.shape[0])[:, np.newaxis]], axis=1)
        if input_data is not None:
            X = np.concatenate([X, input_data], axis=1)
        T = Z.shape[0]
        assert Omegas.shape[0] == T, (Omegas.shape, T)
        if self.nonswitching:
            return self._sample_r(Z, D, X, Omegas, input=input_data)(0)[np.newaxis]
        with Pool(num_workers, len(self.states)) as pool:
            out = pool.map(self._sample_r(Z, D, X, Omegas, input=input_data), self.states)
        return np.stack(out, axis=0)

    def update(self, Z, X, D, Omegas, switches, inputs=None):
        assert not np.any(np.isnan(np.concatenate(Z))), f"Z: {Z}"
        assert not np.any(np.isnan(np.concatenate(X))), f"X: {X}"
        assert not np.any(np.isnan(np.concatenate(D))), f"D: {D}"
        assert not np.any(np.isnan(np.concatenate(Omegas))), f"Omegas: {Omegas}"

        self.R = self.sample_params(Z, X, D, Omegas, switches=switches, input=inputs)
        if not self.nonswitching:
            assert len(self.R) == len(self.states), self.R

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(self.R, "duration_R", step=step, commit=commit, sync=sync)
        log_wandb_scalar_or_array(self.R_mu, "duration_R_mu", step=step, commit=commit, sync=sync)
        log_wandb_scalar_or_array(self.R_sigma, "duration_R_sigma", step=step, commit=commit, sync=sync)

    def get_message(self, Zs, Ds, Omegas, epsilons, switches, inputs=None):
        # Validate input shapes
        T = len(Zs)
        assert len(Ds) == T, f"Ds length mismatch: expected {T}, got {len(Ds)}"
        assert len(Omegas) == T, f"Omegas length mismatch: expected {T}, got {len(Omegas)}"
        assert len(switches) == T, f"switches length mismatch: expected {T}, got {len(switches)}"

        Zs = Zs.astype(int)

        # Select Omegas for each timestep's state
        # Omegas shape: (T, K, D_max-1), we want to select based on state at each timestep
        if self.nonswitching or Omegas.shape[1] == 1:
            # For nonswitching, use first state's Omegas for all
            Omegas_selected = Omegas[:, 0][switches]
        else:
            # Build indices for state-based selection
            # indices shape must be (T, 1, 1) to select along axis 1
            indices = Zs[:, np.newaxis, np.newaxis]
            # Take along axis 1 (state dimension): (T, K, D_max-1) -> (T, 1, D_max-1)
            # Then squeeze axis 1 and filter by switches
            Omegas_selected = np.take_along_axis(Omegas, indices, axis=1).squeeze(axis=1)[switches]

        # Filter all arrays by switches
        shifted_Ds = Ds[switches].copy() - 1
        Zs_filtered = Zs[switches].copy()
        if self.nonswitching:
            Zs_filtered = np.zeros_like(Zs_filtered)

        num_observations = shifted_Ds.shape[0]
        upper_limit = self.D_max - 1

        # Use broadcasting instead of np.repeat for efficiency
        indices = np.arange(upper_limit)
        shifted_col = shifted_Ds[:, np.newaxis]
        ge_mask = (shifted_col >= indices).astype(float)
        kappa = (shifted_col == indices).astype(float) - 0.5 * ge_mask

        # Compute mu using the R matrix and kappa
        mu = (self.R[Zs_filtered, :self.D, :] @ kappa[:, :, np.newaxis]).squeeze()

        # Update Omegas with the duration shifts
        Omegas_masked = Omegas_selected * ge_mask

        # Initialize intercept
        intercept = np.zeros((num_observations, self.R.shape[2], 1))

        # Add affine term if applicable
        if self.affine:
            intercept += self.R[Zs_filtered, self.D, :, np.newaxis]

        # Add inputs if present
        if inputs is not None:
            inputs_filtered = inputs[switches]
            intercept += self.R[Zs_filtered, -self.input_dim:, :] @ inputs_filtered.reshape((-1, self.input_dim, 1))

        # Update mu with the intercept
        mu += - ((Omegas_masked[:, np.newaxis] * self.R[Zs_filtered, :self.D, :]) @ intercept).squeeze()

        # Compute covariance
        cov = (Omegas_masked[:, np.newaxis] * self.R[Zs_filtered, :self.D, :]) @ self.R[Zs_filtered, :self.D, :].transpose((0, 2, 1))

        # Initialize result arrays
        res_mu = np.zeros((T, self.D))
        res_cov = np.zeros((T, self.D, self.D))

        # Directly assign computed values at switch positions
        res_mu[switches] = mu
        res_cov[switches] = cov

        assert mu.shape[-1] == cov.shape[-1], (mu, cov)
        return res_mu, res_cov

    def sample_Omegas(self, Zs, Xs, Ds, inputs=None):
        Z = Zs
        X = Xs
        
        # Add affine term if needed
        if self.affine and Xs.shape[-1] == self.D:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            
        # Add inputs if present
        if inputs is not None:
            X = np.concatenate([X, inputs], axis=1)
            
        T = Z.shape[0]
        X = X[:T]
        
        # Compute Vs based on nonswitching flag
        # Output Vs should always be (T, K, D_max-1)
        if self.nonswitching:
            # Use R[0] for all states, then broadcast to K states
            Vs = (X[:, np.newaxis, :] @ self.R[0])  # Shape: (T, 1, D_max-1)
            Vs = np.repeat(Vs, self.K, axis=1)  # Shape: (T, K, D_max-1)
        else:
            # Compute Vs for ALL K states, not just the current state
            # self.R has shape (K, D_, D_max-1)
            # X has shape (T, D_)
            # We need Vs of shape (T, K, D_max-1)
            Vs = (X[:, np.newaxis, np.newaxis, :] @ self.R[np.newaxis, :, :, :]).squeeze(axis=-2)
            # Result: (T, 1, 1, D_) @ (1, K, D_, D_max-1) -> (T, K, 1, D_max-1) -> squeeze -> (T, K, D_max-1)
            
        try:
            # Sample from Polya-Gamma
            res = random_polyagamma(np.ones(Vs.shape), Vs)
            
            # Ensure proper shape
            if len(res.shape) < 3:
                res = res[:, :, np.newaxis]
                
            if self.nonswitching:
                res[:, 1:] = res[:, 0, np.newaxis, :]

            # Validate output shape - should always be (T, K, D_max-1)
            assert res.shape == (T, self.K, self.D_max - 1), \
                f"sample_Omegas shape error: expected {(T, self.K, self.D_max - 1)}, got {res.shape}"
            return res
        except Exception as e:
            ic(Vs)
            raise e

    def init_pg(self, size):
        shape = (size, self.K, self.D_max - 1)
        result = random_polyagamma(size=shape)
        assert result.shape == shape, f"init_pg shape error: expected {shape}, got {result.shape}"
        return result

    def support(self, *args, **kwargs):
        return 1, self.D_max

    def sample_p(self, i, x):
        print("Sample p used in categorical distribution")
        return x
