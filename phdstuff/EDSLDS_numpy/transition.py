import logging
import numpy as np
import os
import pickle
from icecream import ic
from .utils import worker_pool as Pool
from polyagamma import random_polyagamma
from scipy.stats import multinomial, dirichlet, multivariate_normal
from typing import List

from phdstuff.utils import log_wandb_scalar_or_array
from .utils import sigmoid, invert, turn_matrix_positive_semidefinite, aggresively_turn_matrix_positive_semidefinite, super_aggresively_turn_matrix_positive_semidefinite

# Import Numba kernels for batch transition optimization
try:
    from .numba_kernels import batch_stick_breaking_logprobs, NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False

log = logging.getLogger('transition')

def categorical(alpha):
    return multinomial.rvs(1, alpha / np.sum(alpha)).nonzero()[0][0]


class NonRecurrentTransition:
    def __init__(self, D):
        self.D = D
        self.affine = False

    def sample_Omegas(self, Zs, Xs, input=None):
        return np.zeros_like(Zs).reshape((-1, 1))

    def get_message(self, Zs, Omegas, switches, inputs=None):
        # if len(switches) == len(Zs):
        #     switches = switches[:-1]
        N = len(Zs) + 1
        return np.zeros((N, self.D)), np.zeros((N, self.D, self.D))

    def init_pg(self, size):
        return random_polyagamma(size=size)

    def rotate(self, rotation):
        pass


class Transition(NonRecurrentTransition):
    def __init__(self, K, D=None, A=None):
        super().__init__(D)
        self.alpha = [np.array([1.0 / (K - 1) for i in range(K)]) for j in range(K)]
        for i in range(K):
            self.alpha[i][i] = 0
        self.K = K
        self.loopy = False
        self.states = range(K)
        if A is None:
            A = np.ones((K, K)) / (K - 1)
            np.fill_diagonal(A, 0.0)
        self.A = A

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
            "A": self.A,
            "alpha": self.alpha
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_tran.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{Transition.__name__}_{name}_tran.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        res = Transition(A=kwargs["A"], K=kwargs["K"])
        res.alpha = kwargs["alpha"]
        return res

    def loglikelihood(self, i, j):
        assert i in self.states
        # assert j in self.states
        # assert i != j
        return np.log(self.A[i, j])

    def sample_x(self, i):
        try:
            return categorical(self.A[i].flatten())
        except:
            print(self.A[i].flatten())
            raise

    def sample_A(self, Zs):

        n = dict([(i, dict([(j, 0) for j in self.states])) for i in self.states])

        for Z in Zs:
            X = [z[0] for z in Z]
            now = X[0]
            for x in X:
                if now != x:
                    n[now][x] += 1
                    now = x

        A = np.zeros((self.K, self.K))

        for i in self.states:
            # A[i] = (dirichlet.rvs(self.alpha[i] + np.array(list(n[i].values()))))
            indices = np.repeat(True, self.K)
            indices[i] = False
            A[i, indices] = (dirichlet.rvs((self.alpha[i] + np.array(list(n[i].values())))[indices]))

        log.debug('sampled A:\n%s' % A)

        return A

    def update(self, Z):
        self.A = self.sample_A(Z)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(self.A, "transition_matrix", step=step, commit=commit, sync=sync)


class LoopyTransition(NonRecurrentTransition):
    def __init__(self, K, D=None, A=None):
        super().__init__(D)
        self.alpha = [np.array([1.0 / (K) for i in range(K)]) for j in range(K)]
        self.K = K
        self.loopy = True
        self.states = range(K)
        # A = A + 0.0001
        # for i in self.states:
        #    A[i] = A[i]/A[i].sum()
        if A is None:
            A = np.ones((K, K)) / (K)
        self.A = A

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
            "A": self.A,
            "alpha": self.alpha
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_tran.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_tran.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        res = Transition(A=kwargs["A"], K=kwargs["K"])
        res.alpha = kwargs["alpha"]
        return res

    def loglikelihood(self, i, j):
        # assert i in self.states
        # assert j in self.states
        # assert i != j
        return np.log(self.A[i, j])

    def sample_x(self, i):
        try:
            return categorical(self.A[i].flatten())
        except:
            print(self.A[i].flatten())
            raise

    def sample_A(self, Zs):

        n = dict([(i, dict([(j, 0) for j in self.states])) for i in self.states])

        for Z in Zs:
            now, now_dur = Z[0]
            for (x, d) in Z:
                if now_dur <= d:
                    n[now][x] += 1
                    now = x
                now_dur = d

        A = np.zeros((self.K, self.K))

        for i in self.states:
            A[i] = (dirichlet.rvs(self.alpha[i] + np.array(list(n[i].values()))))

        log.debug('sampled A:\n%s' % A)

        return A

    def update(self, Z):
        self.A = self.sample_A(Z)

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(self.A, "transition_matrix", step=step, commit=commit, sync=sync)


class RecurrentTransition:
    def __init__(self, K: int, D: int, R=None, R_mu=None, R_sigma=None, alpha=None, loopy=False, eps=4.0,
                 affine=False, nonswitching=True, input_dim=None) -> None:
        if alpha is None:
            if not loopy:
                alpha = [np.array([1.0 / (K - 1) for i in range(K)]) for j in range(K)]
                for i in range(K):
                    alpha[i][i] = 0
            else:
                alpha = [np.array([1.0 / (K) for i in range(K)]) for j in range(K)]
        self.affine = affine
        self.alpha = alpha
        self.K = K
        self.D = D
        self.loopy = loopy
        self.eps = eps
        self.states = range(K)
        self.nonswitching = nonswitching
        upper_limit = K - 1 if loopy else K - 2
        D_ = D if not affine else D + 1
        D_ = D_ if input_dim is None else input_dim + D_
        if R is None:
            if nonswitching:
                R = np.ones((1, D_, upper_limit))
            else:
                R = np.stack([np.ones((D_, upper_limit)) for _ in self.states], axis=0)
        self.R = R
        if R_mu is None:
            if nonswitching:
                R_mu = np.zeros((1, D_, upper_limit))
            else:
                R_mu = np.stack([np.zeros((D_, upper_limit)) for _ in self.states], axis=0)
        self.R_mu = R_mu
        if R_sigma is None:
            if nonswitching:
                R_sigma = np.tile(np.eye(D_), (upper_limit, 1, 1)).transpose((1, 2, 0)) * eps
                R_sigma = R_sigma[np.newaxis]
            else:
                R_sigma = np.stack(
                    [np.tile(np.eye(D_), (upper_limit, 1, 1)).transpose((1, 2, 0)) * eps for _ in self.states], axis=0)
        self.R_sigma = R_sigma
        self.input_dim = input_dim

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
            "alpha": self.alpha,
            "R": self.R,
            "D": self.D,
            "R_mu": self.R_mu,
            "R_sigma": self.R_sigma,
            "loopy": self.loopy,
            "affine": self.affine,
            "nonswitching": self.nonswitching,
            "input_dim": self.input_dim,
            "eps": self.eps,
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_rec_tran.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_rec_tran.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        return RecurrentTransition(**kwargs)

    def get_log_transition(self, i, X, omega=None, input=None):
        # omega stays disabled: the omega branch below degenerates to a uniform
        # distribution for single-x calls (builtin max() on a length-1 leading
        # axis subtracts potentials from themselves), which destroys discrete
        # state sampling. All recorded ICDM results (e.g. RSLDS new_nascar_10
        # at 0.90 aligned accuracy) were produced with omega disabled. Briefly
        # enabling it (commit 3986ce1c, 2026-03) collapsed segmentation to
        # near-chance.
        omega = None
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
        if (self.K == 2) and (not self.loopy):
            res = np.zeros((X.shape[0], 2))
            res[:, i] = np.log(0.)
            return res.squeeze()
        if omega is None:
            if len(R.shape) == 3:
                X_ = (X[:, np.newaxis, :] @ R).squeeze()
            else:
                X_ = X @ R
            upper_limit = self.K - 1 if self.loopy else self.K - 2
            p = np.zeros((X_.shape[0], upper_limit + 1))
            mask = (np.eye(upper_limit) * 2 - np.tril(np.ones(upper_limit)))
            p_ = sigmoid(mask[np.newaxis, :, :] * X_[:, np.newaxis, :])
            p_ = np.log(p_)
            p_ = np.tril(p_)
            p_ = np.sum(p_, axis=-1)  # TODO: check if proper axes
            p[:, :-1] = p_
            # ic(sigmoid(-X_))
            p[:, -1] = np.sum(np.log(sigmoid(-X_)), axis=-1)
            assert p.shape[0] == X.shape[0]
            # assert p.shape[-1] == self.K - 1
        else:
            if len(omega.shape) == 2:
                omega = omega[np.newaxis, :, :]
            if len(R.shape) == 3:
                X_ = (X[:, np.newaxis, :] @ R).squeeze()
            else:
                X_ = X @ R
            upper_limit = self.K - 1 if self.loopy else self.K - 2
            states0 = np.arange(upper_limit)
            states1 = np.arange(upper_limit + 1)
            Omega_ = omega[:, i]
            assert Omega_.shape == X_.shape, (Omega_.shape, X_.shape)
            Kappa = (states1[:, np.newaxis] == states0[np.newaxis]).astype(float) - 0.5 * (
                        states1[:, np.newaxis] >= states0[np.newaxis]).astype(float)
            potentials = Kappa[np.newaxis] * X_[:, np.newaxis, :] - 0.5 * (
                        (states1[np.newaxis, :, np.newaxis] >= states0[np.newaxis, np.newaxis]) * (Omega_ * X_ * X_)[:,
                                                                                                  np.newaxis, :])
            potentials = potentials - max(potentials)
            potentials = np.exp(np.sum(potentials, axis=-1))
            if np.all(potentials == 0.):
                potentials = np.ones_like(potentials)
            p = (np.log(potentials) - np.log(np.sum(potentials, axis=1))).reshape((X_.shape[0], -1))
        if not self.loopy:
            res = np.zeros((p.shape[0], p.shape[1] + 1)) + np.log(0.)  # np.inf
        else:
            res = np.zeros((p.shape[0], p.shape[1])) + np.log(0.)
        mask = np.ones(res.shape, dtype=bool)
        if not self.loopy:
            mask[:, i] = False
            res[:, :i] = p[:, :i]
            if i + 1 < res.shape[1]:
                res[:, i + 1:] = p[:, i:]
        else:
            res = p
        # np.putmask(res, mask, p)
        assert np.all(np.isclose(np.sum(np.exp(res.squeeze()), axis=-1),
                                 np.ones_like(np.sum(np.exp(res.squeeze()), axis=-1)))), np.exp(res.squeeze())
        return res.squeeze()

    def get_log_transition_batch(self, X_batch, inputs=None):
        """Batch compute log transition probabilities for all states at once.

        This is an optimized version that computes transition probabilities
        for a batch of latent states, avoiding Python loop overhead.

        Parameters
        ----------
        X_batch : np.ndarray (T, D)
            Batch of latent states
        inputs : np.ndarray (T, input_dim), optional
            External inputs

        Returns
        -------
        np.ndarray (T, K, K)
            Log transition probabilities log P(z'=j | z=i, x) for each timestep
        """
        T = X_batch.shape[0]

        # Handle affine case
        if self.affine and (X_batch.shape[1] == self.D):
            X_batch = np.concatenate([X_batch, np.ones((T, 1))], axis=1)

        # Handle external inputs
        if inputs is not None:
            X_batch = np.concatenate([X_batch, inputs], axis=1)

        # Use Numba kernel if available
        if NUMBA_AVAILABLE and hasattr(self, 'R'):
            R = self.R.astype(np.float64)
            X_batch = X_batch.astype(np.float64)
            return batch_stick_breaking_logprobs(X_batch, R, self.K, self.loopy)

        # Pure NumPy fallback
        result = np.zeros((T, self.K, self.K))
        for i in range(self.K):
            result[:, i, :] = self.get_log_transition(i, X_batch, omega=None, input=None)

        return result

    def loglikelihood(self, x, i, j, omega=None, input = None):
        assert i in self.states
        assert j in self.states
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if self.affine and (x.shape[1] == self.D):
            x = np.concatenate([x, np.ones(x.shape[0])[:, np.newaxis]], axis=1)
        if input is not None:
            x = np.concatenate([x, input], axis=1)
        return self.get_log_transition(i, x, omega)[j]

    def rotate(self, rotation):
        self.R[:, :self.D] = rotation[np.newaxis, :, :] @ self.R[:, :self.D]
        if self.affine:
            self.R[:, -1] = (rotation[np.newaxis, :, :] @ self.R[:, -1]).squeeze()

    def sample_x(self, i, X, omega=None, input=None):
        p = self.get_log_transition(i, X, omega, input=input)
        p = np.exp(p)
        if len(p.shape) > 1:
            p = p.reshape(-1)
        try:
            return categorical(p)
        except Exception:
            print(p)
            p = p - np.finfo(np.float32).epsneg
            p = np.absolute(p)
            return categorical(p)

    def _sample_r(self, Zs, nextZs, Xs, Omegas):
        def _res(k):
            if self.nonswitching:
                indices = np.ones_like(Zs).astype(bool)
            else:
                indices = Zs == k
            upper_limit = self.K - 2 if not self.loopy else self.K - 1
            Kappas = np.zeros((np.sum(indices), upper_limit))
            nextZs_ = nextZs.copy()[indices]
            if not self.loopy:
                nextZs_[nextZs_ > k] = nextZs_[nextZs_ > k] - 1
            Xs_ = Xs[indices]
            Omegas_ = Omegas[indices]
            # ic(Omegas_.shape)
            if len(Xs_) == 0:
                return self.R[k]
            R = self.R[k].copy()
            for j in range(upper_limit):
                Kappas[:, j] = (nextZs_ == j).astype(float) - 0.5 * (nextZs_ >= j).astype(float)
                Omegas_j = Omegas_[:, j] * (nextZs_ >= j).astype(float)
                mu0 = self.R_mu[k][:, j]
                precision0 = invert(self.R_sigma[k][:, :, j])
                theta0 = precision0 @ mu0
                Xs_prime = np.sqrt(Omegas_j[:, np.newaxis]) * Xs_
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

    def sample_Omegas(self, Zs, Xs, input = None):
        Z = Zs if isinstance(Zs, np.ndarray) else np.array(Zs)
        X = Xs
        if self.affine and Xs.shape[-1] == self.D:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        if input is not None:
            X = np.concatenate([X, input], axis=1)
        # Vs = self.R[Z] @ X
        T = Z.shape[0]
        assert len(X) == T, (T, X.shape)
        X = X[:T]
        if self.nonswitching:
            Vs = (X[:, np.newaxis, :] @ self.R[np.zeros_like(Z)])  # .squeeze()
            Vs = np.repeat(Vs, self.K, axis=1)
        else:
            Vs = ((X[:, np.newaxis, np.newaxis, :] @ self.R[np.newaxis])).squeeze()
        try:
            res = random_polyagamma(np.ones(Vs.shape), Vs)
            if len(res.shape) < 3:
                res = res[:, :, np.newaxis]
            if self.nonswitching:
                res[:, 1:] = res[:, 0, np.newaxis, :]
            return res
        except Exception as e:
            # ic(mask)
            ic(Vs)
            raise e

    def sample_params(self, Zs: List[np.ndarray], Xs: List[np.ndarray], Omegas: List[np.ndarray], switches,
                      num_workers=32, inputs = None
                      ):
        try:
            Omegas = np.concatenate(
                [np.take_along_axis(Omega_[:-1], Zs_[:-1, np.newaxis, np.newaxis], axis=1).squeeze(axis=1)[switches_[1:]] for
                 Omega_, Zs_, switches_ in zip(Omegas, Zs, switches)])
        except Exception:
            Omegas = np.concatenate(
                [np.take_along_axis(Omega_[:-1], Zs_[:-1, np.newaxis, np.newaxis], axis=1).squeeze(axis=1)[switches_][1:] for
                 Omega_, Zs_, switches_ in zip(Omegas, Zs, switches)])

        Z = np.concatenate([Zs_[:-1][switches_[1:]] for Zs_, switches_ in zip(Zs, switches)])  # TODO: sprawdzic indeksy
        X = np.concatenate([Xs_[:-1][switches_[1:]] for Xs_, switches_ in zip(Xs, switches)])
        Z1 = np.concatenate([Zs_[1:][switches_[1:]] for Zs_, switches_ in zip(Zs, switches)])
        input = None if inputs is None else np.concatenate([inputs[1:][switches_[1:]] for switches_ in switches])
        if self.affine and (X.shape[1] == self.D):
            X = np.concatenate([X, np.ones(X.shape[0])[:, np.newaxis]], axis=1)
        if input is not None:
            X = np.concatenate([X, input], axis=1)
        T = Z.shape[0]
        assert Omegas.shape[0] == T, (Omegas.shape, T)
        if self.nonswitching:
            return self._sample_r(Z, Z1, X, Omegas)(0)[np.newaxis]
        with Pool(num_workers, len(self.states)) as pool:
            out = pool.map(self._sample_r(Z, Z1, X, Omegas), self.states)
        return np.stack(out, axis=0)

    def update(self, Z, X, Omegas, switches, inputs=None):
        self.R = self.sample_params(Z, X, Omegas, switches=switches, inputs=inputs)
        if not self.nonswitching:
            assert len(self.R) == len(self.states), self.R

    def log_to_wandb(self, step=None, commit=None, sync=None):
        log_wandb_scalar_or_array(self.R, "transition_R", step=step, commit=commit, sync=sync)

    def get_message(self, Zs, Omegas, switches, inputs=None):
        #TODO zwarunkować po inpucie
        Zs = Zs.astype(int)
        indices = Zs[:-1, np.newaxis, np.newaxis] if not self.nonswitching else np.zeros_like(
            Zs[:-1, np.newaxis, np.newaxis])
        try:
            Omegas = np.take_along_axis(Omegas[:-1], indices, axis=1).squeeze(axis=1)[switches[1:]]
        except Exception as e:
            raise e
        shifted_Zs = Zs[1:][switches[1:]].copy()
        Zs = Zs[:-1][switches[1:]].copy()
        if self.nonswitching:
            Zs = np.zeros_like(Zs)
        if not self.loopy:
            shifted_Zs[shifted_Zs > Zs] = shifted_Zs[shifted_Zs > Zs] - 1
        num_switches = shifted_Zs.shape[0]
        upper_limit = self.K - 2 if not self.loopy else self.K - 1
        # Use broadcasting instead of np.repeat for efficiency
        indices = np.arange(upper_limit)
        shifted_col = shifted_Zs[:, np.newaxis]
        ge_mask = (shifted_col >= indices).astype(float)
        kappa = (shifted_col == indices).astype(float) - 0.5 * ge_mask
        mu = (self.R[Zs, :self.D, :] @ kappa[:, :, np.newaxis]).squeeze()
        Omegas = Omegas * ge_mask
        intercept = np.zeros((num_switches, self.R.shape[2], 1))
        if self.affine:
            intercept += self.R[Zs, self.D, :, np.newaxis]
        if inputs is not None:
            intercept += self.R[Zs, -self.input_dim:, :]@inputs[switches[1:]].reshape((-1, self.D, 1))
        mu += - ((Omegas[:, np.newaxis] * self.R[Zs, :self.D, :]) @ intercept).squeeze()
        cov = (Omegas[:, np.newaxis] * self.R[Zs, :self.D, :]) @ self.R[Zs, :self.D, :].transpose((0, 2, 1))
        res_mu = np.zeros((switches.shape[0], mu.shape[1]))
        res_cov = np.zeros((switches.shape[0], cov.shape[1], cov.shape[2]))
        tmp_mu = np.zeros((mu.shape[0] + 1, mu.shape[1]))
        tmp_cov = np.zeros((cov.shape[0] + 1, cov.shape[1], cov.shape[2]))
        tmp_mu[1:] = mu
        tmp_cov[1:] = cov
        res_mu[switches] = tmp_mu
        res_cov[switches] = tmp_cov
        assert mu.shape[-1] == cov.shape[-1], (mu, cov)
        return res_mu, res_cov

    def init_pg(self, size):
        upper_limit = self.K - 2 if not self.loopy else self.K - 1
        size = (size, self.K, upper_limit)
        return random_polyagamma(size=size)


class HDPTransition(NonRecurrentTransition):
    def __init__(self, L, D, alpha=1.0, gamma=1.0, kappa=1.0,
                 alpha_a0=1.0, alpha_b0=1.0,
                 gamma_a0=1.0, gamma_b0=1.0, alpha0_init=0.0001, eps = 1e-6):
        """
        Inicjalizacja przejścia HDP.
        
        Args:
            L (int): Liczba stanów
            alpha (float): Parametr koncentracji dla procesu Dirichleta pierwszego poziomu
            gamma (float): Parametr koncentracji dla procesu bazowego
            kappa (float): Parametr self-transition bias
            alpha_a0, alpha_b0 (float): Parametry prior'a gamma dla alpha
            gamma_a0, gamma_b0 (float): Parametry prior'a gamma dla gamma
        """
        super().__init__(D)
        self.affine = False
        self.alpha0_init = alpha0_init
        self.L = L
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.states = np.arange(L)
        
        # Parametry prior'ów
        self.alpha0_a_pri = alpha_a0
        self.alpha0_b_pri = alpha_b0
        self.gamma0_a_pri = gamma_a0
        self.gamma0_b_pri = gamma_b0
        
        # Inicjalizacja parametrów
        self.beta = np.ones(L) / L
        self.pi = np.random.dirichlet(self.alpha * self.beta, size=L)

        # Statystyki wystarczające
        # self.state_counts = np.zeros((L, L), dtype=int)
        self.m = np.zeros((L, L), dtype=int)
        self.eps = eps  # Mała wartość do unikania dzielenia przez zero

    def sample_alpha(self, alpha0, m_mat, n_mat, n_ft):
        """
        Próbkuje parametr koncentracji alpha używając metody auxiliary variable.
        """
        r_vec = []
        tmp = n_mat.sum(axis=1)
        for val in tmp:
            if val > 0:
                r_vec.append(np.random.beta(alpha0 + 1, val))
        r_vec = np.array(r_vec)
        s_vec = np.random.binomial(1, n_mat.sum(axis=1) / (n_mat.sum(axis=1) + alpha0))
        alpha0 = np.random.gamma(self.alpha0_a_pri + (m_mat.sum()) - sum(s_vec),
                       1 / (self.alpha0_b_pri - sum(np.log(r_vec + self.eps))))  ## not consider first time point

        ## sample alpha_init
        nper = n_ft.sum()
        eta = np.random.beta(self.alpha0_init + 1, nper)
        ntab = self.m_init.sum()
        pi_m = (self.alpha0_a_pri + ntab - 1) / (self.alpha0_a_pri + ntab - 1 + nper * (self.alpha0_b_pri - np.log(eta + self.eps)))
        indicator = np.random.binomial(1, pi_m)
        if indicator:
            alpha0_init = np.random.gamma(self.alpha0_a_pri + ntab, 1 / (self.alpha0_b_pri - np.log(eta + self.eps)))
        else:
            alpha0_init = np.random.gamma(self.alpha0_a_pri + ntab - 1, 1 / (self.alpha0_b_pri - np.log(eta + self.eps)))

        return alpha0, alpha0_init  # , r_vec, s_vec

    def sample_gamma(self, K, m_mat, m_init, gamma0):  ## first time point will affect gamma

        num_tabs = m_mat.sum() + m_init.sum()
        eta = np.random.beta(gamma0 + 1, num_tabs)

        pi_m = (self.gamma0_a_pri + K - 1) / (self.gamma0_a_pri + K - 1 + num_tabs * (self.gamma0_b_pri - np.log(eta + self.eps)))
        indicator = np.random.binomial(1, pi_m)

        if indicator:
            gamma0 = np.random.gamma(self.gamma0_a_pri + K, 1 / (self.gamma0_b_pri - np.log(eta + self.eps)))
        else:
            gamma0 = np.random.gamma(self.gamma0_a_pri + K - 1, 1 / (self.gamma0_b_pri - np.log(eta + self.eps)))

        return gamma0  # , eta

    def sample_m(self, n_mat, n_ft, beta_vec, alpha0, alpha0_init):
        L = n_mat.shape[0]
        m_mat = np.zeros((L, L))

        for j in range(L):
            for k in range(L):
                if n_mat[j, k] == 0:
                    m_mat[j, k] = 0
                else:
                    x_vec = np.random.binomial(1, alpha0 * beta_vec[k] / (np.arange(n_mat[j, k]) + alpha0 * beta_vec[k]))
                    x_vec = np.array(x_vec).reshape(-1)
                    m_mat[j, k] = sum(x_vec)

        m_init = np.zeros(L)
        for j in range(L):
            if n_ft[j] == 0:
                m_init[j] = 0
            else:
                x_vec = np.random.binomial(1, alpha0_init * beta_vec[j] / (np.arange(n_ft[j]) + alpha0_init * beta_vec[j]))
                x_vec = np.array(x_vec).reshape(-1)
                m_init[j] = sum(x_vec)

        return m_mat, m_init

    def sample_beta(self, m_mat, m_init, gamma0):  ## first time point will affect beta, gamma
        L = m_mat.shape[0]
        prob_vec = m_mat.sum(axis=0) + (gamma0 / L) + m_init
        prob_vec[prob_vec < 0.01] = 0.01
        beta_vec = dirichlet.rvs(prob_vec.squeeze()).squeeze()
        return beta_vec

    def sample_pi(self, n_mat, n_ft, alpha0, beta_vec):  ## first time point won't affect pi_bar
        L = n_mat.shape[0]
        pi_bar = np.zeros((L, L))
        for k in range(L):
            prob_vec = (alpha0 * beta_vec) + n_mat[k]
            prob_vec[prob_vec < 0.01] = 0.01
            pi_bar[k] = dirichlet.rvs(prob_vec.squeeze()).squeeze()

        prob_vec = (self.alpha0_init * beta_vec) + n_ft
        prob_vec[prob_vec < 0.01] = 0.01
        pi_init = dirichlet.rvs(prob_vec.squeeze()).squeeze()
        return pi_bar, pi_init

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
            "pi": self.pi,
            "alpha": self.alpha
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_tran.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{HDPTransition.__name__}_{name}_tran.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        res = Transition(A=kwargs["A"], K=kwargs["K"])
        res.alpha = kwargs["alpha"]
        return res

    def loglikelihood(self, i, j):
        assert i in self.states
        # assert j in self.states
        # assert i != j
        return np.log(self.pi[i, j])

    def sample_x(self, i):
        try:
            return categorical(self.pi[i].flatten())
        except:
            print(self.pi[i].flatten())
            raise



    def update(self, Z):
        """
        Próbkuje wszystkie parametry modelu.
        """
        n_mat = np.zeros((self.L, self.L), dtype=int)
        n_ft = np.zeros(self.L, dtype=int)
        for z in Z:
            for i in range(len(z) - 1):
                n_mat[z[i][0], z[i + 1][0]] += 1
            n_ft[z[0][0]] += 1
        # Próbkowanie pomocniczych zmiennych
        self.m, self.m_init = self.sample_m(n_mat, n_ft, self.beta, self.alpha, self.alpha0_init)
        
        # Próbkowanie miary bazowej beta
        self.beta = self.sample_beta(self.m, self.m_init, self.gamma)
        self.pi, self.pi_init = self.sample_pi(n_mat, n_ft, self.alpha, self.beta)
        # Próbkowanie hiperparametrów
        self.alpha, self.alpha0_init= self.sample_alpha(self.alpha, self.m, n_mat, n_ft)
        self.gamma = self.sample_gamma(self.L, self.m, self.m_init, self.gamma)


class StickyHDPTransition(HDPTransition):
    def __init__(self, L, D, alpha=1.0, gamma=1.0, kappa=1.0,
                 alpha_a0=1.0, alpha_b0=1.0, c_pri=1.0, d_pri=1.0,
                 gamma_a0=1.0, gamma_b0=1.0, alpha0_init=0.0001, eps=1e-6):
        """
        Inicjalizacja przejścia Sticky HDP-HMM.

        Args:
            L (int): Liczba stanów
            alpha (float): Parametr koncentracji dla procesu Dirichleta pierwszego poziomu
            gamma (float): Parametr koncentracji dla procesu bazowego
            kappa (float): Parametr self-transition bias
            alpha_a0, alpha_b0 (float): Parametry prior'a gamma dla alpha
            c_pri, d_pri (float): Parametry prior'a beta dla stick_ratio
            gamma_a0, gamma_b0 (float): Parametry prior'a gamma dla gamma
        """
        super().__init__(L, D, alpha, gamma, kappa, alpha_a0, alpha_b0, gamma_a0, gamma_b0, alpha0_init, eps)

        # Parametry dla sticky HDP-HMM
        self.kappa = kappa
        self.c_pri = c_pri
        self.d_pri = d_pri
        self.stick_ratio = np.random.beta(c_pri, d_pri)
        self.concentration = self.alpha + self.kappa
        self.rho0 = self.concentration * self.stick_ratio
        self.alpha0 = self.concentration - self.rho0

        # Dodatkowe statystyki dla sticky HDP-HMM
        self.w = np.zeros(L, dtype=int)
        self.m_bar = np.zeros((L, L), dtype=int)

    def transform(self, concentration, stick_ratio):
        """
        Transformuje concentration i stick_ratio na parametry rho0 i alpha0.
        """
        rho0 = concentration * stick_ratio
        alpha0 = concentration - rho0
        return rho0, alpha0

    def sample_stick_ratio(self, w_vec, m_mat):
        """
        Próbkuje stick_ratio dla sticky HDP-HMM.
        """
        stick_ratio = np.random.beta(w_vec.sum() + self.c_pri, m_mat.sum() - w_vec.sum() + self.d_pri)
        return stick_ratio

    def sample_m_w_mbar(self, n_mat, n_ft, beta_vec, alpha0, alpha0_init, rho0):
        """
        Próbkuje pomocnicze zmienne m, w oraz m_bar dla sticky HDP-HMM.
        """
        L = n_mat.shape[0]
        # Próbkowanie m
        m_mat = np.zeros((L, L))
        for j in range(L):
            for k in range(L):
                if n_mat[j, k] == 0:
                    m_mat[j, k] = 0
                else:
                    x_vec = np.random.binomial(1, (alpha0 * beta_vec[k] + rho0 * (j == k)) /
                                               (np.arange(n_mat[j, k]) + alpha0 * beta_vec[k] + rho0 * (j == k)))
                    m_mat[j, k] = sum(x_vec)

        w_vec = np.zeros(L)
        m_mat_bar = m_mat.copy()
        # Próbkowanie w jeśli rho > 0
        if rho0 > 0:
            stick_ratio = rho0 / (rho0 + alpha0)
            for j in range(L):
                if m_mat[j, j] > 0:
                    w_vec[j] = np.random.binomial(m_mat[j, j],
                                                  stick_ratio / (stick_ratio + beta_vec[j] * (1 - stick_ratio)))
                    m_mat_bar[j, j] = m_mat[j, j] - w_vec[j]

        # Dla pierwszego punktu czasowego
        m_init = np.zeros(L)
        for j in range(L):
            if n_ft[j] == 0:
                m_init[j] = 0
            else:
                x_vec = np.random.binomial(1,
                                           alpha0_init * beta_vec[j] / (np.arange(n_ft[j]) + alpha0_init * beta_vec[j]))
                m_init[j] = sum(x_vec)

        return m_mat, m_init, w_vec, m_mat_bar

    def sample_concentration(self, m_mat, n_mat, alpha0, rho0, m_init, n_ft, alpha0_init):
        """
        Próbkuje parametr koncentracji dla sticky HDP-HMM.
        """
        r_vec = []
        tmp = n_mat.sum(axis=1)
        concentration = alpha0 + rho0

        for val in tmp:
            if val > 0:
                r_vec.append(np.random.beta(concentration + 1, val))
        r_vec = np.array(r_vec).reshape(-1)

        s_vec = np.random.binomial(1, n_mat.sum(axis=1) / (n_mat.sum(axis=1) + concentration))
        s_vec = np.array(s_vec).reshape(-1)

        concentration = np.random.gamma(self.alpha0_a_pri + (m_mat.sum()) - sum(s_vec),
                                        1 / (self.alpha0_b_pri - sum(np.log(r_vec + self.eps))))

        # Próbkowanie alpha_init
        nper = n_ft.sum()
        eta = np.random.beta(alpha0_init + 1, nper)
        ntab = m_init.sum()
        pi_m = (self.alpha0_a_pri + ntab - 1) / (
                    self.alpha0_a_pri + ntab - 1 + nper * (self.alpha0_b_pri - np.log(eta + self.eps)))
        indicator = np.random.binomial(1, pi_m)
        if indicator:
            alpha0_init = np.random.gamma(self.alpha0_a_pri + ntab, 1 / (self.alpha0_b_pri - np.log(eta + self.eps)))
        else:
            alpha0_init = np.random.gamma(self.alpha0_a_pri + ntab - 1,
                                          1 / (self.alpha0_b_pri - np.log(eta + self.eps)))

        return concentration, alpha0_init

    def sample_pi(self, n_mat, n_ft, alpha0, rho0, beta_vec):
        """
        Próbkowanie prawdopodobieństw przejścia dla sticky HDP-HMM.
        """
        L = n_mat.shape[0]
        pi_bar = np.zeros((L, L))
        for k in range(L):
            prob_vec = (alpha0 * beta_vec) + n_mat[k]
            prob_vec[k] += rho0
            prob_vec[prob_vec < 0.01] = 0.01
            pi_bar[k] = np.random.dirichlet(prob_vec)

        prob_vec = (self.alpha0_init * beta_vec) + n_ft
        prob_vec[prob_vec < 0.01] = 0.01
        pi_init = np.random.dirichlet(prob_vec)
        return pi_bar, pi_init

    def update(self, Z):
        """
        Próbkuje wszystkie parametry modelu dla sticky HDP-HMM.
        """
        n_mat = np.zeros((self.L, self.L), dtype=int)
        n_ft = np.zeros(self.L, dtype=int)
        for z in Z:
            for i in range(len(z) - 1):
                n_mat[z[i][0], z[i + 1][0]] += 1
            n_ft[z[0][0]] += 1

        # Próbkowanie pomocniczych zmiennych dla sticky HDP-HMM
        self.m, self.m_init, self.w, self.m_bar = self.sample_m_w_mbar(
            n_mat, n_ft, self.beta, self.alpha0, self.alpha0_init, self.rho0)

        # Próbkowanie miary bazowej beta
        self.beta = self.sample_beta(self.m_bar, self.m_init, self.gamma)

        # Próbkowanie prawdopodobieństw przejścia
        self.pi, self.pi_init = self.sample_pi(n_mat, n_ft, self.alpha0, self.rho0, self.beta)

        # Próbkowanie hiperparametrów
        self.concentration, self.alpha0_init = self.sample_concentration(
            self.m, n_mat, self.alpha0, self.rho0, self.m_init, n_ft, self.alpha0_init)

        self.gamma = self.sample_gamma(len(self.m_bar), self.m_bar, self.m_init, self.gamma)
        self.stick_ratio = self.sample_stick_ratio(self.w, self.m)
        self.rho0, self.alpha0 = self.transform(self.concentration, self.stick_ratio)
