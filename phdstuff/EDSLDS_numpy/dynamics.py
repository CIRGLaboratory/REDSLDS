import logging
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from scipy.stats import invwishart, multivariate_normal, matrix_normal

from phdstuff.utils import log_wandb_scalar_or_array
from .utils import bayesian_linear_regression_posterior, invert, make_positive_definite, \
    turn_matrix_positive_semidefinite, aggresively_turn_matrix_positive_semidefinite, \
    super_aggresively_turn_matrix_positive_semidefinite


def _invwishart_pd(nu, scale):
    """invwishart.rvs, falling back to a positive-definite floor of the scale matrix."""
    try:
        return invwishart.rvs(nu, scale)
    except np.linalg.LinAlgError:
        log.warning("dynamics: inverse-Wishart scale floored to positive definite")
        return invwishart.rvs(nu, make_positive_definite(scale))


def _matrix_normal_pd(mean, rowcov, colcov):
    """matrix_normal.rvs, falling back to positive-definite floors of both covariances."""
    try:
        return matrix_normal.rvs(mean, rowcov, colcov)
    except np.linalg.LinAlgError:
        log.warning("dynamics: matrix-normal covariances floored to positive definite")
        return matrix_normal.rvs(mean, make_positive_definite(rowcov), make_positive_definite(colcov))
from typing import Optional
log = logging.getLogger('dynamics')

log_2_pi = np.log(2 * np.pi)


def mvnormal(mu, tau):
    # return tfd.MultivariateNormalFullCovariance(mu, tau).sample()
    return multivariate_normal.rvs(mean=mu, cov=tau)


class AbstractDynamics(ABC):
    @abstractmethod
    def loglikelihood(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_obs(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_D(self):
        pass


class LinearGaussianDynamics(AbstractDynamics):
    def __init__(self, As=None, bs=None, Sigmas=None, mus_0=None, Sigmas_0=None, nu_0=None, V_0=None, Lambda_0=None,
                 B_0=None,
                 K=None, x_dim=None, eps=0.1, affine=False, input_dim=None):
        """
        Initializes the Linear Gaussian Dynamics.
        Args:
            As: The transition matrices.
            bs: The intercept terms.
            Sigmas: The covariance matrices.
            mus_0: The prior means.
            Sigmas_0: The prior covariance matrices.
            nu_0: The prior degrees of freedom.
            V_0: The prior scale matrices.
            Lambda_0: The prior precision matrices.
            B_0: The prior scale matrices for the intercept terms.
            K: The number of states.
            x_dim: The dimension of the dynamics matrices.
            eps: The epsilon value.
            affine: Whether the dynamics matrices are affine.
            input_dim: The dimension of the inputs to the system.
        """
        if As is None:
            As = np.stack([
                np.eye(x_dim) + np.random.randn(x_dim, x_dim) for _ in range(K)
            ])
        else:
            K = len(As)
            x_dim = As[0].shape[0]
        if bs is None and affine:
            bs = np.ones((K, x_dim))
        if affine and (As.shape[-1] == x_dim):
            temp = np.zeros((K, x_dim, x_dim + 1))
            temp[:, :, :-1] = As
            temp[:, :, -1] = bs
            As = temp
        self.x_dim = x_dim
        if Sigmas is None:
            Sigmas = np.stack([
                np.eye(x_dim) for _ in range(K)
            ]) * eps
        if mus_0 is None:
            mus_0 = np.zeros((K, x_dim))
        if Sigmas_0 is None:
            Sigmas_0 = np.stack([np.eye(x_dim) for _ in range(K)]) * eps * 100000
        if nu_0 is None:
            nu_0 = np.array([x_dim + 2] * K)
            if affine:
                nu_0 += 1
        if V_0 is None:
            V_0 = np.stack([np.eye(x_dim) for _ in range(K)]) * eps * 1000
        if Lambda_0 is None:
            Lambda_0 = np.stack([invert(A) for A in Sigmas_0])
        if affine and Lambda_0.shape[1] == x_dim:
            temp = np.zeros((K, x_dim + 1, x_dim + 1))
            temp[:, :-1, :-1] = Lambda_0
            temp[:, -1, -1] = temp[:, 0, 0]
            Lambda_0 = temp
        if B_0 is None:
            B_0 = 0.99 * np.stack([np.eye(x_dim) for _ in range(K)])
            # B_0 = np.zeros((K, x_dim, x_dim))
        if affine and B_0.shape[1] == x_dim:
            temp = np.zeros((K, x_dim + 1, x_dim))
            temp[:, :-1, :] = B_0
            B_0 = temp
        self.params = {
            "mus_0": mus_0,
            "Sigmas_0": Sigmas_0,
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
        self.As = As
        self.K = As.shape[0]
        self.affine = affine
        self.states = np.arange(As.shape[0])
        self.input_dim = input_dim
        self.draw_stats = {"rejected_A_draws": 0, "skipped_A_updates": 0}   # of the last update()

    # Stability-truncated dynamics draw: the MNIW posterior of A is truncated to a spectral radius of at
    # most RHO_MAX (rejection sampling). Under the near-flat mixed prior (dyn_V_0 = 1e-8) a state can draw
    # a slightly explosive A with a tiny noise, and a long segment in that state then grows geometrically
    # in the Kalman draw. Healthy states stay below a radius of 1.18 (174 final dumps of the rebuttal
    # matrix), so the truncation is inactive on a healthy chain; the RNG stream is untouched unless a
    # proposal is rejected. When every proposal is rejected the state keeps its previous A for this sweep
    # (the prior mean if that one is explosive as well); the sidecar counts both events.
    RHO_MAX = 1.2
    A_DRAW_RETRIES = 10

    def spectral_radius(self, A):
        return float(np.abs(np.linalg.eigvals(np.asarray(A)[:, :self.x_dim])).max())

    def __copy__(self):
        copy_res = LinearGaussianDynamics(As=copy(self.As),
                                          Sigmas=copy(self.Sigmas),
                                          mus_0=copy(self.params["mus_0"]),
                                          Sigmas_0=copy(self.params["Sigmas_0"]),
                                          nu_0=copy(self.params["nu_0"]),
                                          V_0=copy(self.params["V_0"]),
                                          Lambda_0=copy(self.params["Lambda_0"]),
                                          B_0=copy(self.params["B_0"]),
                                          K=copy(self.K),
                                          x_dim=copy(self.x_dim),
                                          affine=self.affine)
        copy_res.params["nu_n"] = copy(self.params["nu_n"])
        copy_res.params["V_n"] = copy(self.params["V_n"])
        copy_res.params["Lambda_n"] = copy(self.params["Lambda_n"])
        copy_res.params["B_n"] = copy(self.params["B_n"])
        return copy_res

    def __deepcopy__(self, memodict={}):
        copy_res = LinearGaussianDynamics(As=deepcopy(self.As, memodict),
                                          Sigmas=deepcopy(self.Sigmas, memodict),
                                          mus_0=deepcopy(self.params["mus_0"], memodict),
                                          Sigmas_0=deepcopy(self.params["Sigmas_0"], memodict),
                                          nu_0=deepcopy(self.params["nu_0"], memodict),
                                          V_0=deepcopy(self.params["V_0"], memodict),
                                          Lambda_0=deepcopy(self.params["Lambda_0"], memodict),
                                          B_0=deepcopy(self.params["B_0"], memodict),
                                          K=deepcopy(self.K, memodict),
                                          x_dim=deepcopy(self.x_dim, memodict),
                                          affine=self.affine)
        memodict[id(self)] = copy_res
        copy_res.params["nu_n"] = deepcopy(self.params["nu_n"], memodict)
        copy_res.params["V_n"] = deepcopy(self.params["V_n"], memodict)
        copy_res.params["Lambda_n"] = deepcopy(self.params["Lambda_n"], memodict)
        copy_res.params["B_n"] = deepcopy(self.params["B_n"], memodict)
        return copy_res

    def dump(self, path: str, name: str) -> None:
        """
        Dumps the dynamics matrices to the given path.
        Args:
            path: The path to dump the matrices to.
            name: The name of the matrices.
        """
        dump_dic = {
            "params": self.params,
            "Sigmas": self.Sigmas,
            "As": self.As,
            "affine": self.affine
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_dyn.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)

    def get_D(self) -> int:
        """
        Returns the dimension of the dynamics matrices.
        Returns: The dimension of the dynamics matrices.

        """
        return self.x_dim

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_dyn.pickle"), "rb") as ifile:
            init_dic = pickle.load(ifile)
        params = init_dic["params"]
        kwargs = params.copy()
        kwargs.pop("nu_n")
        kwargs.pop("V_n")
        kwargs.pop("Lambda_n")
        kwargs.pop("B_n")
        kwargs["Sigmas"] = init_dic["Sigmas"]
        kwargs["As"] = init_dic["As"]
        kwargs["affine"] = init_dic["affine"]
        res = LinearGaussianDynamics(**kwargs)
        res.params = params
        return res

    def rotate(self, rotation):
        """
        Rotates the dynamics matrices by the given rotation matrix.
        Args:
            rotation: The rotation matrix to apply to the dynamics matrices.
        """
        self.As[:, :self.x_dim, :self.x_dim] = rotation[np.newaxis, :, :] @ self.As[:, :self.x_dim,
                                                                            :self.x_dim] @ rotation[np.newaxis, :,
                                                                                           :].transpose((0, 2, 1))
        if self.affine:
            self.As[:, :, -1] = (rotation[np.newaxis, :, :] @ self.As[:, :, -1, np.newaxis]).squeeze()
        if self.input_dim is not None:
            left = self.x_dim if not self.affine else self.x_dim + 1
            self.As[:, :self.x_dim, left:] = rotation[np.newaxis, :, :] @ self.As[:, :self.x_dim,
                                                                          left:] @ rotation[np.newaxis, :,
                                                                                   :].transpose((0, 2, 1))

    def rotate_samples(self, rotation, samples):
        """
        Rotates the given samples using the provided rotation matrix.

        Parameters
        ----------
        rotation : numpy.ndarray
            The rotation matrix to apply to the samples.
        samples : numpy.ndarray
            The samples to be rotated.

        Returns
        -------
        numpy.ndarray
            The rotated samples.
        """
        samples = samples @ rotation.T
        return samples

    def loglikelihood(self, state, previous_observation, observation, inputs=None):
        """
        Computes the log-likelihood of the given observation given the previous observation and the state.
        Args:
            state: The state to compute the log-likelihood for.
            previous_observation: The previous observation.
            observation: The observation to compute the log-likelihood for.
            inputs: The inputs to the system.

        Returns: The log-likelihood of the observation given the previous observation and the state.

        """
        if self.affine and previous_observation.shape[-1] == self.x_dim:
            previous_observation = np.concatenate([previous_observation, np.ones(1)])
        if inputs is not None:
            previous_observation = np.concatenate([previous_observation, inputs])
        # ic(previous_observation)
        mu = self.As[state] @ previous_observation
        Sigma = self.Sigmas[state]
        # return tfd.MultivariateNormalFullCovariance(mu, Sigma).log_prob(observation)
        return multivariate_normal.logpdf(observation, mean=mu, cov=Sigma, allow_singular=True)

    def loglikelihood_batch(self, previous_observations, observations, inputs=None):
        """
        Computes log-likelihoods for ALL states at once, for a sequence of observations.

        Args:
            previous_observations: (T, D) array of previous latent states
            observations: (T, D) array of current latent states
            inputs: Optional (T, input_dim) array of inputs

        Returns:
            (K, T) array of log-likelihoods for each state and timestep
        """
        T = observations.shape[0]
        D = self.x_dim

        # Handle affine case
        if self.affine and previous_observations.shape[-1] == self.x_dim:
            prev_obs = np.concatenate([previous_observations, np.ones((T, 1))], axis=1)
        else:
            prev_obs = previous_observations

        if inputs is not None:
            prev_obs = np.concatenate([prev_obs, inputs], axis=1)

        # Compute means for all states: mu[k, t] = As[k] @ prev_obs[t]
        # As shape: (K, D, D') where D' = D or D+1 (affine) or D+1+input_dim
        # prev_obs shape: (T, D')
        # Result shape: (K, T, D)
        means = np.einsum('kij,tj->kti', self.As, prev_obs)

        # Compute differences: obs[t] - mu[k, t]
        # observations shape: (T, D), means shape: (K, T, D)
        diff = observations[np.newaxis, :, :] - means  # (K, T, D)

        # Pre-compute Sigma inverses and log-determinants if not cached
        Sigma_invs = np.linalg.inv(self.Sigmas)  # (K, D, D)
        _, log_dets = np.linalg.slogdet(self.Sigmas)  # (K,)

        # Mahalanobis distance: diff @ Sigma_inv @ diff.T for each (k, t)
        # einsum: 'kti,kij,ktj->kt' means sum over i,j for each k,t
        mahal = np.einsum('kti,kij,ktj->kt', diff, Sigma_invs, diff)

        # Log probability: -0.5 * (D*log(2π) + log|Σ| + mahalanobis)
        log_probs = -0.5 * (D * log_2_pi + log_dets[:, np.newaxis] + mahal)

        return log_probs

    def sample_obs(self, state, previous_observation=None, inputs=None):
        """
        Samples an observation given the previous observation and the state.
        Args:
            state: The state to sample the observation from.
            previous_observation: The previous observation.
            inputs: The inputs to the system.

        Returns: The sampled observation.
        """
        assert state in self.states, (state, self.states)
        if previous_observation is not None:
            if self.affine and previous_observation.shape[-1] == self.x_dim:
                previous_observation = np.concatenate([previous_observation, np.ones(1)])
            if inputs is not None:
                previous_observation = np.concatenate([previous_observation, inputs])
            mu = self.As[state] @ previous_observation  # + self.bs[state]
            Sigma = self.Sigmas[state]
        else:
            mu = self.params["mus_0"][state]
            Sigma = self.params["Sigmas_0"][state]
        res = mvnormal(mu, Sigma)
        if len(res.shape) == 1:
            return res[:self.x_dim]
        return res[:, :self.x_dim]

    def update(self, Zs, Xs, inputs=None):
        """
        Updates the dynamics matrices given the observations and the latent states.
        Args:
            Zs: The latent states.
            Xs: The observations.
            inputs: The inputs to the system.
        """
        Zs = np.concatenate(Zs)
        prevXs = np.concatenate([x[:-1] for x in Xs])
        Xs = np.concatenate([x[1:] for x in Xs])
        if self.affine and prevXs.shape[1] == self.x_dim:
            prevXs = np.concatenate([prevXs, np.ones(prevXs.shape[0])[:, np.newaxis]], axis=1)
        if inputs is not None:
            inputs = np.concatenate(inputs)
            prevXs = np.concatenate([prevXs, inputs], axis=1)
        self.params = bayesian_linear_regression_posterior(self.params, Zs, prevXs, Xs, self.K)
        rejected, skipped = 0, 0
        for i in range(self.K):
            V_n = self.params["V_n"][i]
            try:
                self.Sigmas[i] = invwishart.rvs(self.params["nu_n"][i], V_n)
            except:
                try:
                    V_n = turn_matrix_positive_semidefinite(V_n)
                    self.Sigmas[i] = invwishart.rvs(self.params["nu_n"][i], V_n)
                except np.linalg.LinAlgError:
                    try:
                        V_n = aggresively_turn_matrix_positive_semidefinite(V_n)
                        self.Sigmas[i] = invwishart.rvs(self.params["nu_n"][i], V_n)
                    except np.linalg.LinAlgError:
                        V_n = super_aggresively_turn_matrix_positive_semidefinite(V_n)
                        self.Sigmas[i] = _invwishart_pd(self.params["nu_n"][i], V_n)
            lam_inv = invert(self.params["Lambda_n"][i])
            previous = self.As[i].copy()
            for _ in range(self.A_DRAW_RETRIES + 1):
                self.As[i] = self._draw_A(i, lam_inv)
                if self.spectral_radius(self.As[i]) <= self.RHO_MAX:
                    break
                rejected += 1
            else:
                skipped += 1
                keep = self.spectral_radius(previous) <= self.RHO_MAX
                self.As[i] = previous if keep else self.params["B_0"][i].T
                log.warning("dynamics: state %d (%d steps) drew a spectral radius > %.2f in %d proposals; %s",
                            i, self.params["nu_n"][i] - self.params["nu_0"][i], self.RHO_MAX,
                            self.A_DRAW_RETRIES + 1, "previous A kept" if keep else "prior mean used")
            log.debug(f'sampled A for state {i}: {self.As[i]}')
            log.debug(f'sampled dynamics sigma {i}: {self.Sigmas[i]}')
        self.draw_stats = {"rejected_A_draws": rejected, "skipped_A_updates": skipped}

    def _draw_A(self, i, lam_inv):
        """One matrix-normal proposal of A for state i, repairing the covariances as far as needed."""
        try:
            lam_inv = turn_matrix_positive_semidefinite(lam_inv)
            self.Sigmas[i] = turn_matrix_positive_semidefinite(self.Sigmas[i])
            return matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T
        except np.linalg.LinAlgError:
            try:
                self.Sigmas[i] = turn_matrix_positive_semidefinite(self.Sigmas[i])
                for k, v in self.params.items():
                    print(f"{k}: {v}")
                lam_inv = turn_matrix_positive_semidefinite(lam_inv)
                return matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T
            except np.linalg.LinAlgError:
                try:
                    self.Sigmas[i] = aggresively_turn_matrix_positive_semidefinite(self.Sigmas[i])
                    lam_inv = aggresively_turn_matrix_positive_semidefinite(lam_inv)
                    return matrix_normal.rvs(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T
                except np.linalg.LinAlgError:
                    self.Sigmas[i] = super_aggresively_turn_matrix_positive_semidefinite(self.Sigmas[i])
                    lam_inv = super_aggresively_turn_matrix_positive_semidefinite(lam_inv)
                    return _matrix_normal_pd(self.params["B_n"][i], lam_inv, self.Sigmas[i]).T

    def get_intercept_theta(self, Zs, inputs=None):
        """
        Computes the intercept term for the given latent states.
        Args:
            Zs: The latent states.
            inputs: The inputs to the system.

        Returns: The intercept term for the given latent states.
        """
        if not self.affine and inputs is None:
            return np.zeros((Zs.shape[0], self.x_dim))
        result = self.As[Zs, :, self.x_dim]
        if inputs is not None:
            result += inputs @ self.As[Zs, :, -self.input_dim:]
        return -result

    def get_transitions(self, states=None, with_intercept=False):
        """
        Returns the transition matrices for the given states.
        Args:
            states: The states to return the transition matrices for.
            with_intercept: Whether to include the intercept term in the transition matrices.

        Returns: The transition matrices for the given states.

        """
        As = self.As
        if self.affine and not with_intercept:
            As = As[:, :, :self.x_dim]
        if As.shape[2] > self.x_dim:
            As = As[:, :, :(self.x_dim + 1)]
        if states is None:
            return As
        return As[states]

    def get_sigmas(self, states: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the covariance matrices for the given states.
        Args:
            states: The states to return the covariance matrices for.

        Returns: The covariance matrices for the given states.
        """
        if states is None:
            return self.Sigmas + np.eye(self.x_dim)[np.newaxis] * 1.e-6
        return self.Sigmas[states] + np.eye(self.x_dim)[np.newaxis] * 1.e-6

    def log_to_wandb(self, step=None, commit=None, sync=None):
        """
        Logs the dynamics matrices to wandb.
        Args:
            step: The step to log the matrices at.
            commit: Whether to commit the matrices.
            sync: Whether to sync the matrices.
        """
        for k, v in self.params.items():
            if "0" in k:
                continue
            log_wandb_scalar_or_array(v, f"dynamics_{k}", step=step, commit=commit, sync=sync)
