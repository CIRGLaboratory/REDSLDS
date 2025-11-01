from abc import ABC
from abc import abstractmethod
import logging
import os
import pickle

import numpy as np
import scipy
from scipy.stats import invwishart
from scipy.stats import matrix_normal
from scipy.stats import multivariate_normal

from phdstuff.utils import log_wandb_scalar_or_array

from .utils import aggresively_turn_matrix_positive_semidefinite
from .utils import bayesian_linear_regression_posterior
from .utils import invert
from .utils import turn_matrix_positive_semidefinite

log = logging.getLogger('emissions')

log_2_pi = np.log(2 * np.pi)


def invwishart_(nu, L):
    nu = np.maximum(nu, L.shape[0] + 2)
    return invwishart.rvs(nu, L)


def mvnormal(mu, sigma):
    return multivariate_normal.rvs(mean=mu, cov=sigma)


class AbstractEmission(ABC):
    @abstractmethod
    def loglikelihood(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_obs(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class LinearGaussianEmission(AbstractEmission):
    def __init__(self, Cs=None, Sigmas=None, mu_0=None, Sigma_0=None, nu_0=None, V_0=None, Lambda_0=None, B_0=None,
                 obs_dim=None, x_dim=None, K=None, eps=0.1, input_dim=None, affine=False, bs=None):
        self.affine = affine
        if Cs is not None:
            K = len(Cs)
            obs_dim, x_dim_ = Cs[0].shape
            # If affine and Cs already includes bias column, x_dim is one less
            if affine and x_dim is None:
                x_dim = x_dim_ - 1
            elif x_dim is None:
                x_dim = x_dim_
        else:
            Cs = np.stack([
                np.eye(obs_dim, x_dim) + np.random.randn(obs_dim, x_dim) * eps for _ in range(K)
            ])
        # Handle affine: expand Cs to include bias column
        if affine and Cs.shape[2] == x_dim:
            if bs is None:
                bs = np.zeros((K, obs_dim))
            temp = np.zeros((K, obs_dim, x_dim + 1))
            temp[:, :, :-1] = Cs
            temp[:, :, -1] = bs
            Cs = temp
        self.K, self.obs_dim, self.x_dim = K, obs_dim, x_dim
        if Sigmas is None:
            Sigmas = np.stack([
                np.eye(obs_dim) for _ in range(K)
            ]) * eps
        if mu_0 is None:
            mu_0 = np.zeros((K, obs_dim))
        if Sigma_0 is None:
            Sigma_0 = np.stack([
                np.eye(obs_dim) for _ in range(K)
            ]) * eps
        if nu_0 is None:
            nu_0 = np.array([obs_dim + 2] * K)
        if V_0 is None:
            V_0 = np.stack([
                np.eye(obs_dim) for _ in range(K)
            ])
        if Lambda_0 is None:
            Lambda_0 = np.stack([
                np.eye(x_dim) for _ in range(K)
            ])
        if B_0 is None:
            B_0 = np.zeros((K, x_dim, obs_dim))
        self.params = {
            "mu_0": mu_0,
            "Sigma_0": Sigma_0,
            "nu_0": nu_0,
            "nu_n": nu_0.copy(),
            "V_0": V_0,
            "V_n": V_0.copy(),
            "Lambda_0": Lambda_0,
            "Lambda_n": Lambda_0.copy(),
            "B_0": B_0,
            "B_n": B_0.copy()
        }
        self.Sigmas = Sigmas
        self.Cs = Cs
        self.K = Cs.shape[0]
        self.states = np.arange(Cs.shape[0])
        self.input_dim = None

    def dump(self, path, name):
        dump_dic = {
            "params": self.params,
            "Sigmas": self.Sigmas,
            "Cs": self.Cs,
            "input_dim": self.input_dim
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_emi.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    def get_intercept_theta(self, Zs):
        if not self.affine:
            return np.zeros((Zs.shape[0], self.x_dim))
        return -self.Cs[Zs, :, :-1] @ invert(self.Sigmas[Zs, :, :]) @ self.Cs[Zs, :, -1]

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_emi.pickle"), "rb") as ifile:
            init_dic = pickle.load(ifile)
        params = init_dic["params"]
        kwargs = params.copy()
        kwargs.pop("nu_n")
        kwargs.pop("V_n")
        kwargs.pop("Lambda_n")
        kwargs.pop("B_n")
        kwargs["Sigmas"] = init_dic["Sigmas"]
        kwargs["Cs"] = init_dic["Cs"]
        res = LinearGaussianEmission(**kwargs)
        res.params = params
        return res

    def loglikelihood(self, state, dynamics, observation):
        mu = self.Cs[state] @ dynamics  # + self.ds[state]
        Sigma = self.Sigmas[state]
        return multivariate_normal.logpdf(observation, mu, Sigma)

    def loglikelihood_batch(self, dynamics, observations):
        """
        Computes log-likelihoods for ALL states at once, for a sequence of observations.

        Args:
            dynamics: (T, D) array of latent states
            observations: (T, obs_dim) array of observations

        Returns:
            (K, T) array of log-likelihoods for each state and timestep
        """
        obs_dim = self.obs_dim

        # Compute means for all states: mu[k, t] = Cs[k] @ dynamics[t]
        # Cs shape: (K, obs_dim, x_dim) or (K, obs_dim, x_dim+1) if affine
        # dynamics shape: (T, D)
        # Result shape: (K, T, obs_dim)
        Cs = self.Cs[:, :, :dynamics.shape[1]]  # Handle affine case
        means = np.einsum('kij,tj->kti', Cs, dynamics)

        # Add affine intercept if present
        if getattr(self, 'affine', False) and dynamics.shape[1] == self.x_dim:
            means = means + self.Cs[:, :, -1][:, np.newaxis, :]  # (K, 1, obs_dim) broadcast

        # Compute differences: obs[t] - mu[k, t]
        diff = observations[np.newaxis, :, :] - means  # (K, T, obs_dim)

        # Pre-compute Sigma inverses and log-determinants
        Sigma_invs = np.linalg.inv(self.Sigmas)  # (K, obs_dim, obs_dim)
        _, log_dets = np.linalg.slogdet(self.Sigmas)  # (K,)

        # Mahalanobis distance
        mahal = np.einsum('kti,kij,ktj->kt', diff, Sigma_invs, diff)

        # Log probability
        log_2_pi = np.log(2 * np.pi)
        log_probs = -0.5 * (obs_dim * log_2_pi + log_dets[:, np.newaxis] + mahal)

        return log_probs

    def sample_obs(self, state, dynamics):
        assert state in self.states, (state, self.states)
        if dynamics is not None:
            mu = self.Cs[state] @ dynamics  # + self.ds[state]
            Sigma = self.Sigmas[state]
        else:
            mu = self.params["mu_0"][state]
            Sigma = self.params["Sigma_0"][state]
        return mvnormal(mu, Sigma)

    def generate_centers(self, states, dynamics):  # TODO: Speed it up
        res = []
        for Z, X in zip(states, dynamics):
            tmp = []
            for i, z in enumerate(Z):
                x = X[i]
                tmp.append(self.Cs[z] @ x)
            res.append(np.stack(tmp))
        return res

    def get_transitions(self, states=None, with_intercept=False):
        Cs = self.Cs
        if self.affine and not with_intercept:
            Cs = Cs[:, :, :-1]
        if states is None:
            return Cs
        return Cs[states]

    def get_sigmas(self, states=None):
        if states is None:
            return self.Sigmas + np.eye(self.obs_dim)[np.newaxis] * 1.e-6
        return self.Sigmas[states] + np.eye(self.obs_dim)[np.newaxis] * 1.e-6

    def update(self, Zs, Xs, Ys):
        Zs = np.concatenate(Zs)
        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        # non_na = ~np.any(np.isnan(Ys), axis=1)
        # Zs = Zs[non_na]
        # Xs = Xs[non_na]
        # Ys = Ys[non_na]

        self.params = bayesian_linear_regression_posterior(self.params, Zs, Xs, Ys,
                                                           self.K)
        for i in range(self.K):
            self.Sigmas[i] = invwishart_(self.params["nu_n"][i], self.params["V_n"][i])
            lam_inv = invert(self.params["Lambda_n"][i])
            try:
                self.Cs[i] = matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T
            except np.linalg.LinAlgError:
                try:
                    self.Sigmas[i] = turn_matrix_positive_semidefinite(self.Sigmas[i])
                    lam_inv = turn_matrix_positive_semidefinite(lam_inv)
                    self.Cs[i] = matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T
                except np.linalg.LinAlgError:
                    self.Sigmas[i] = aggresively_turn_matrix_positive_semidefinite(self.Sigmas[i])
                    lam_inv = aggresively_turn_matrix_positive_semidefinite(lam_inv)
                    self.Cs[i] = matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T
            #
            # self.Cs[i] = matrix_normal.rvs(self.params["B_n"][i],
            #                                             np.linalg.inv(self.params["Lambda_n"][i]), self.Sigmas[i]).T
            log.debug(f'sampled C for state {i}: {self.Cs[i]}')
            log.debug(f'sampled emission sigma {i}: {self.Sigmas[i]}')

    def log_to_wandb(self, step=None, commit=None, sync=None):
        for k, v in self.params.items():
            if "0" in k:
                continue
            log_wandb_scalar_or_array(v, f"emission_{k}", step=step, commit=commit, sync=sync)


class NonswitchingLinearGaussianEmission(AbstractEmission):
    def __init__(self, K, C=None, Sigma=None, mu_0=None, Sigma_0=None, nu_0=None, V_0=None, Lambda_0=None, B_0=None,
                 eps=0.1, obs_dim=None, x_dim=None, updatable=True, bs=None, affine=False, input_dim=None):
        self.affine = affine
        if bs is None and affine:
            bs = np.zeros((obs_dim))
        if C is not None:
            obs_dim, x_dim = C.shape
        else:
            C = np.eye(obs_dim, x_dim)
        if affine:
            temp = np.zeros((obs_dim, x_dim + 1))
            temp[:, :-1] = C
            temp[:, -1] = bs
            C = temp
        if input_dim is not None:
            temp = np.ones((obs_dim, C.shape[1] + input_dim))
            temp[:, :C.shape[1]] = C
            C = temp
        self.K, self.obs_dim, self.x_dim = K, obs_dim, x_dim
        if Sigma is None:
            Sigma = np.eye(obs_dim) * eps
        if mu_0 is None:
            mu_0 = np.zeros((1, obs_dim))
        if Sigma_0 is None:
            Sigma_0 = np.eye(obs_dim) * eps
        if nu_0 is None:
            nu_0 = np.array([obs_dim + 2])
        if affine:
            nu_0 += 1
        if V_0 is None:
            V_0 = np.eye(obs_dim)

        if Lambda_0 is None:
            Lambda_0 = np.eye(x_dim)
        if C.shape[1] > x_dim and Lambda_0.shape[0] < C.shape[1]:
            temp = np.eye(C.shape[1])
            temp[:x_dim, :x_dim] = Lambda_0
            temp[x_dim:, x_dim:] *= temp[0, 0]
            Lambda_0 = temp
        if B_0 is None:
            B_0 = np.zeros((x_dim, obs_dim))
        if C.shape[1] > x_dim and B_0.shape[0] < C.shape[1]:
            temp = np.zeros((C.shape[1], obs_dim))
            temp[:x_dim, :] = B_0
            B_0 = temp
        x_dim = x_dim if not affine else x_dim + 1
        self.params = {
            "mu_0": mu_0,
            "Sigma_0": Sigma_0,
            "nu_0": nu_0,
            "nu_n": nu_0.copy(),
            "V_0": V_0.reshape((1, obs_dim, obs_dim)),
            "V_n": V_0.reshape((1, obs_dim, obs_dim)).copy(),
            "Lambda_0": Lambda_0.reshape((1, x_dim, x_dim)),
            "Lambda_n": Lambda_0.reshape((1, x_dim, x_dim)).copy(),
            "B_0": B_0.reshape((1, x_dim, obs_dim)),
            "B_n": B_0.reshape((1, x_dim, obs_dim)).copy()
        }
        self.Sigma = Sigma
        self.C = C
        self.bs = bs
        self.K = K
        self.updatable = updatable
        self.input_dim = input_dim

    def dump(self, path, name):
        dump_dic = {
            "params": self.params,
            "Sigma": self.Sigma,
            "C": self.C,
            "K": self.K,
            "updatable": self.updatable,
            "input_dim": self.input_dim
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_emi.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_emi.pickle"), "rb") as ifile:
            init_dic = pickle.load(ifile)
        params = init_dic["params"]
        kwargs = params.copy()
        kwargs.pop("nu_n")
        kwargs.pop("V_n")
        kwargs.pop("Lambda_n")
        kwargs.pop("B_n")
        kwargs["Sigma"] = init_dic["Sigma"]
        kwargs["C"] = init_dic["C"]
        kwargs["updatable"] = init_dic["updatable"]
        kwargs["K"] = init_dic["K"]
        res = NonswitchingLinearGaussianEmission(**kwargs)
        res.params = params
        return res

    def rotate(self):
        # TODO: Check what to do with inputs
        if self.affine:
            upper, orthor = scipy.linalg.rq(self.C[:, :-1])
        else:
            upper, orthor = scipy.linalg.rq(self.C)
        "Constrain minor diagnoal of upper to be positive to avoid sign flipping"
        rotate = np.eye(self.x_dim)
        for j in range(self.x_dim):
            if np.sign(upper[self.obs_dim - self.x_dim + j, j]) < 0:
                rotate[j, j] = -1

        upper = upper @ rotate
        orthor = rotate @ orthor

        # Contrain the emission matrix to be an upper matrix
        if self.affine:
            self.C[:, :-1] = upper
        else:
            self.C = upper
        return orthor

    def normalize(self):
        # TODO: sprawdzić co z inputem
        if self.affine:
            C_temp = self.C[:, :-1]
        else:
            C_temp = self.C
        # Normalize columns
        L = np.diag(C_temp.T @ C_temp)
        L = np.diag(np.power(L, -0.5))
        C_temp = C_temp @ L
        if self.affine:
            self.C[:, :-1] = C_temp
        else:
            self.C = C_temp

    def get_intercept_theta(self, Zs):
        if not self.affine:
            return np.zeros((Zs.shape[0], self.x_dim))
        Zs = np.zeros_like(Zs)
        return (-self.C[np.newaxis, :, :-1][Zs].transpose((0, 2, 1)) @ invert(self.Sigma[np.newaxis, :, :]) @ self.C[
                                                                                                                  np.newaxis][
                                                                                                              Zs, :, -1,
                                                                                                              np.newaxis]).squeeze()

    def loglikelihood(self, state, dynamics, observation, input=None):
        if self.affine and dynamics.shape[-1] == self.x_dim:
            dynamics = np.concatenate([dynamics, np.ones(1)])
        if input is not None:
            dynamics = np.concatenate([dynamics, input])
        mu = self.C @ dynamics  # + self.ds[state]
        Sigma = self.Sigma
        return multivariate_normal.logpdf(observation, mu, Sigma)

    def sample_obs(self, state, dynamics, input=None):
        if dynamics is not None:
            if self.affine and dynamics.shape[-1] == self.x_dim:
                dynamics = np.concatenate([dynamics, np.ones(1)])
            if input is not None:
                dynamics = np.concatenate([dynamics, input])
            mu = self.C @ dynamics  # + self.ds[state]
            Sigma = self.Sigma
        else:
            mu = self.params["mu_0"]
            Sigma = self.params["Sigma_0"]
        res = mvnormal(mu, Sigma)
        if len(res.shape) == 1:
            return res[:self.obs_dim]
        return res[:, :self.obs_dim]

    def generate_centers(self, states, dynamics, input=None):  # TODO: Speed it up
        res = []
        for Z, X in zip(states, dynamics):
            if self.affine and X.shape[-1] == self.x_dim:
                X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            if input is not None:
                X = np.concatenate([X, input], axis=1)
            tmp = []
            for i, z in enumerate(Z):
                x = X[i]
                y = self.C @ x
                if len(y.shape) == 1:
                    y = y[:self.obs_dim]
                else:
                    y = y[:, :self.obs_dim]
                tmp.append(y)
            res.append(np.stack(tmp))
        return res

    def get_transitions(self, states=None, with_intercept=False):
        Cs = np.tile(self.C, (self.K, 1, 1))
        if self.affine and not with_intercept:
            Cs = Cs[:, :, :self.x_dim]
        if states is None:
            return Cs
        return Cs[states]

    def get_sigmas(self, states=None):
        sigmas = np.tile(self.Sigma, (self.K, 1, 1))
        if states is None:
            return sigmas + np.eye(self.obs_dim)[np.newaxis] * 1.e-6
        return sigmas[states] + np.eye(self.obs_dim)[np.newaxis] * 1.e-6

    def update(self, Zs, Xs, Ys, normalize=False):
        if not self.updatable:
            return
        Zs = np.concatenate(Zs)
        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        # non_na = ~np.any(np.isnan(Ys), axis=1)
        # Zs = Zs[non_na]
        # Xs = Xs[non_na]
        # Ys = Ys[non_na]
        if self.affine and Xs.shape[1] == self.x_dim:
            Xs = np.concatenate([Xs, np.ones(Xs.shape[0])[:, np.newaxis]], axis=1)
        
        self.params = bayesian_linear_regression_posterior(self.params, np.zeros(Zs.shape), Xs, Ys, 1)
        i = 0
        self.Sigma = invwishart_(self.params["nu_n"][i], self.params["V_n"][i])
        lam_inv = invert(self.params["Lambda_n"][i])
        try:
            self.C = matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigma).T
        except np.linalg.LinAlgError:
            try:
                self.Sigma = turn_matrix_positive_semidefinite(self.Sigma)
                lam_inv = turn_matrix_positive_semidefinite(lam_inv)
                self.C = matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigma).T
            except np.linalg.LinAlgError:
                self.Sigma = aggresively_turn_matrix_positive_semidefinite(self.Sigma)
                lam_inv = aggresively_turn_matrix_positive_semidefinite(lam_inv)
                self.C = matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigma).T
        if normalize:
            self.normalize()
        log.debug(f'sampled C for state {i}: {self.C}')
        log.debug(f'sampled emission sigma {i}: {self.Sigma}')

    def log_to_wandb(self, step=None, commit=None, sync=None):
        for k, v in self.params.items():
            if "0" in k:
                continue
            log_wandb_scalar_or_array(v, f"emission_{k}", step=step, commit=commit, sync=sync)


class Gaussian(AbstractEmission):
    def __init__(self, nu, Lambda, mu_0, kappa, mu, tau):
        self.nu = nu
        self.Lambda = Lambda
        self.mu_0 = mu_0  # mu_0 is the mean of the prior on the mean
        self.kappa = float(kappa)

        self.mu = mu  # mu is the current value of the mean for each state
        self.tau = tau  # tau is the current precision matrix for each state

        self.states = range(len(mu))
        self.K = len(self.states)

    def loglikelihood(self, state, obs):
        assert state in self.states, (state, self.states)
        return multivariate_normal.logpdf(obs, self.mu[state], np.linalg.inv(self.tau[state]))

    def sample_obs(self, state):
        assert state in self.states, (state, self.states)
        return mvnormal(self.mu[state], np.linalg.inv(self.tau[state]))

    def sample_mean_prec(self, Zs, Ys):

        n = dict([(i, []) for i in self.states])

        for Z, Y in zip(Zs, Ys):
            X = [z[0] for z in Z]
            for t, s in enumerate(X):
                n[s].append(np.array([Y[t]]))

        for i in self.states:
            n[i] = np.array(n[i]).T
            n[i] = np.squeeze(n[i])

        # print n[i]
        # for i in self.states:
        # log.debug("state: %s"%i)
        # log.debug("observations: %s"%n[i].round(2))
        taus, mus = [], []
        for i in self.states:

            if len(n[i]) > 0:
                try:
                    ybar = np.mean(n[i], 1)
                except IndexError:
                    ybar = np.mean(n[i])

                ybar = ybar.flatten()

                S = np.sum(np.array([
                    np.outer((yi - ybar), (yi - ybar))
                    for yi in n[i].T
                ]), 0)
            else:
                # wtf? we don't have any of these observations...
                # fall back on the prior mean and we won't updated
                # the precision
                ybar = np.array(self.mu_0[i])
                S = 0

            # assert not np.isnan(S), S
            # assert not np.isinf(S), S

            # log.debug("ybar[%s]: %s"%(i,ybar))
            mu_n = (
                    (
                            (self.kappa / (self.kappa + len(n[i]))) * self.mu_0[i]
                    ) +
                    (
                            (len(n[i]) / (self.kappa + len(n[i]))) * ybar
                    )
            )
            # log.debug("mu_n[%s]: %s"%(i,mu_n))
            kappa_n = self.kappa + len(n[i])
            nu_n = self.nu + len(n[i])
            Lambda_n = (
                    self.Lambda +
                    S +
                    (
                            (self.kappa * len(n[i])) / (self.kappa + len(n[i])) *
                            (ybar - self.mu_0[i]) * (ybar - self.mu_0[i]).T
                    )
            )

            # if (np.isnan(Lambda_n)).any():
            #     Lambda_n = self.Lambda
            # if np.isnan(nu_n):
            #     nu_n = self.nu

            try:
                sigma = invwishart_(nu_n, (Lambda_n))
            except:
                print(Lambda_n)
                raise
            # form precion matrix
            if isinstance(sigma, float):
                sigma = np.array([[sigma]])
            if len(sigma.shape) < 2:
                sigma = sigma.reshape((1, 1))
            tau = np.linalg.inv(sigma)
            # log.debug("tau[%s]: %s"%(i,tau))
            try:
                tau_scaled = np.linalg.inv(sigma / kappa_n)
            except np.linalg.LingAlgError:
                tau_scaled = 1.0 / (sigma / kappa_n)
            # log.debug("tau_scaled[%s]: %s"%(i,tau_scaled))
            try:
                mu = mvnormal(mu_n, np.linalg.inv(tau_scaled))
            except ValueError:
                mu = self.mu_0[i]
                log.debug('fell back onto the prior mu_0 probably due to lack of observations in this state')
            taus.append(tau)
            mus.append(mu)
            log.debug('sampled obs mean for state %s: %s' % (i, mus[-1]))
            log.debug('sampled obs prec for state %s: %s' % (i, taus[-1]))

        return mus, taus

    def update(self, Z, Y):
        mu, tau = self.sample_mean_prec(Z, Y)
        self.mu = mu
        self.tau = tau


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Z = np.load('Z.npy')
    Y = np.load('Y.npy')
    O = Gaussian(
        nu=1,
        Lambda=np.eye(3),
        mu_0=[0., 0., 0.],
        kappa=0.01,
        mu=[-3., 0., 3.],
        tau=np.array([1., 1., 1.])
    )
    # x = np.linspace(-4,4,100)
    # for i in range(3):
    #    plt.plot(x,[plt.exp(O.likelihood(i,xi)) for xi in x])
    # plt.show()
    # mus, sigmas = O.sample_mean_prec(Z,Y)
    plt.figure()
    for j in range(3):
        mus = np.array([O.sample_mean_prec([Z], [Y])[0][j] for i in range(100)]).flatten()
        plt.hist(mus, alpha=0.5)

    plt.figure()
    for j in range(3):
        taus = np.array([O.sample_mean_prec([Z], [Y])[1][j] for i in range(100)]).flatten()
        plt.hist(taus, alpha=0.5)
    plt.show()
