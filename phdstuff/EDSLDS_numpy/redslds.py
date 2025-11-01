import copy
import json
import numpy as np
import os
import time
import traceback
from copy import deepcopy
from icecream import ic
from matplotlib import pyplot as plt
from phdstuff.EDSLDS_numpy.utils import worker_pool as Pool
from scipy.special import logsumexp, logit
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from typing import List, Optional

from phdstuff.EDSLDS_numpy.duration import RecurrentDuration, DummyDuration, NonRecurrentDuration
from phdstuff.EDSLDS_numpy.dynamics import LinearGaussianDynamics
from phdstuff.EDSLDS_numpy.edhmm import log, Categorical
from phdstuff.EDSLDS_numpy.emission import LinearGaussianEmission
from phdstuff.EDSLDS_numpy.initial import AbstractInitial, LearnableInitial
from phdstuff.EDSLDS_numpy.plot import plot_actual_shifts, plot_observations, \
    plot_phase_portrait, plot_emission, plot_durations
from phdstuff.EDSLDS_numpy.redhmm import REDHMM
from phdstuff.EDSLDS_numpy.transition import RecurrentTransition, LoopyTransition
from phdstuff.EDSLDS_numpy.utils import invert, turn_matrix_positive_semidefinite, \
    aggresively_turn_matrix_positive_semidefinite, fit_decision_list
from phdstuff.EDSLDS_numpy.diagnostics import beam_truncation_stats, beam_truncation_trace, \
    posterior_increments, record_posterior
from phdstuff.plotting import plot_states
from phdstuff.utils import permute, log_wandb_scalar_or_array

# Import Numba kernels for optimized Kalman filtering
try:
    from phdstuff.EDSLDS_numpy.numba_kernels import (
        cholesky_lower, cholesky_mvn_sample,
        NUMBA_AVAILABLE
    )
except ImportError:
    NUMBA_AVAILABLE = False


def _logsumexp(a, b):
    return logsumexp(np.stack([a, b]), axis=0)


def _append_jsonl(path, obj):
    """Best-effort append; metrics I/O must never kill a run."""
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(obj) + "\n")
    except (OSError, TypeError, ValueError):
        log.exception("failed to append sweep metrics to %s", path)


class REDSLDS(object):
    def __init__(self, initial: AbstractInitial, transition: RecurrentTransition, emission: LinearGaussianEmission,
                 dynamics: LinearGaussianDynamics, duration, extended_kalman=False, nobeam=False,
                 ignore_switches=False, forward_discrete_states_sample=False):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.dynamics = dynamics
        self.duration = duration
        self.K = len(initial)
        self.states = range(self.K)
        self.extended_kalman = extended_kalman
        self.nobeam = nobeam
        self.affine = transition.affine or duration.affine or dynamics.affine or emission.affine
        self.ignore_switches = ignore_switches
        self.forward_discrete_states_sample = forward_discrete_states_sample
        self._x_abs_max = np.inf     # divergence bound of the latent trajectory; beam() sets it (see X_SCALE_FACTOR)
        self._validate_discrete_sampler_configuration()

    def _validate_discrete_sampler_configuration(self):
        if self.forward_discrete_states_sample and not self.nobeam:
            raise ValueError(
                "forward_discrete_states_sample=True requires nobeam=True; "
                "set nobeam=True or disable forward sampling"
            )

    @staticmethod
    def _prepend_initial_latent(initial_latent, latents):
        initial_row = np.asarray(initial_latent).reshape(1, -1)
        return np.concatenate([initial_row, latents], axis=0)

    def _backward_kalman_filter(self, obs, states, durations, epsilons=None, omegas_dur=None, omegas_tran=None,
                                skip_duration_message=False, skip_transition_message=False):
        assert isinstance(states, np.ndarray)
        assert isinstance(durations, np.ndarray)
        switches = np.ones_like(durations).astype(bool)
        if not self.ignore_switches:
            switches[1:] = (durations == 1)[:-1]
        # switches = np.zeros(durations.shape[0], dtype=bool)
        # switches[:-1] = durations[:-1] == 1
        states = states.astype(int)
        Cs = self.emission.get_transitions(states)
        eSigmas = self.emission.get_sigmas(states)
        Ds = self.dynamics.get_transitions(states)
        dSigmas = self.dynamics.get_sigmas(states)
        T = states.shape[0]
        dur_thetas = np.zeros((T+1, dSigmas.shape[1]))
        tran_thetas = np.zeros((T+1, dSigmas.shape[1]))
        emi_thetas = np.zeros((T, dSigmas.shape[1]))
        dyn_thetas = np.zeros((T, dSigmas.shape[1]))
        dur_lambdas = np.zeros((T+1, dSigmas.shape[1], dSigmas.shape[2]))
        tran_lambdas = np.zeros((T+1, dSigmas.shape[1], dSigmas.shape[2]))
        if self.dynamics.affine:
            dyn_thetas = self.dynamics.get_intercept_theta(states)
        if self.emission.affine:
            emi_thetas = self.emission.get_intercept_theta(states)
        if not skip_duration_message:
            try:
                dur_thetas, dur_lambdas = self.duration.get_message(states,
                                                                    durations,
                                                                    omegas_dur,
                                                                    epsilons=epsilons,
                                                                    switches=switches)
            except Exception as e:
                print(e)
                ic(states.shape)
                ic(durations.shape)
                ic(omegas_dur.shape)
                ic(epsilons.shape)
                ic(switches.shape)
                ic(states[switches].shape)
                ic(durations[switches].shape)
                ic(omegas_dur[switches].shape)
                ic(epsilons[switches].shape)
                raise e

        assert not np.any(np.isnan(dur_thetas)), f"dur_mus: {dur_thetas}"
        assert not np.any(np.isnan(dur_lambdas)), f"dur_sigmas: {dur_lambdas}"
        if not skip_transition_message:
            tran_thetas, tran_lambdas = self.transition.get_message(states,
                                                                    omegas_tran,
                                                                    switches=switches)
        assert not np.any(np.isnan(tran_thetas)), f"dur_mus: {tran_thetas}"
        assert not np.any(np.isnan(tran_lambdas)), f"tran_sigmas: {tran_lambdas}"

        Lambdas = np.zeros((T+1, dSigmas.shape[1], dSigmas.shape[2]))
        Thetas = np.zeros((T+1, dSigmas.shape[1]))
        # Batch invert all emission and dynamics covariances at once
        R_invs = np.linalg.inv(eSigmas)
        dSigma_invs = np.linalg.inv(dSigmas)
        Lambdas[T] = Cs[T - 1].T @ R_invs[T - 1] @ Cs[T - 1]
        if np.all(obs[T - 1] != np.nan):
            Thetas[T] = Cs[T - 1].T @ R_invs[T - 1] @ obs[T - 1] + emi_thetas[T - 1]  # TODO:?
        for t in range(T - 1, -1, -1):
            # Compute - use pre-computed dSigma inverse
            dSigma_inv_t = dSigma_invs[t]
            C1 = Lambdas[t + 1] + dSigma_inv_t
            C1_inv = invert(C1)
            J = Lambdas[t + 1] @ C1_inv
            L = np.eye(J.shape[0]) - J
            # Predict - reuse dSigma_inv_t
            Lambda_1 = Ds[t].T @ (L @ Lambdas[t + 1] @ L.T + J @ dSigma_inv_t @ J.T) @ Ds[t]
            Lambda_1 = Lambda_1 + dur_lambdas[t] + tran_lambdas[t]
            Theta_1 = Thetas[t + 1]
            if self.dynamics.affine:
                Theta_1 = Theta_1 + Lambdas[t + 1] @ dyn_thetas[t]

            Theta_1 = Ds[t].T @ L @ (Theta_1)
            Theta_1 = Theta_1 + dur_thetas[t] + tran_thetas[t]
            # Update
            if t > 0:
                Lambdas[t] = Lambda_1 + Cs[t - 1].T @ R_invs[t-1] @ Cs[t-1]
                Thetas[t] = Theta_1
                assert not np.any(np.isnan(Lambdas[t])) and not np.any(np.isinf(Lambdas[t])), (t, Lambdas[t], tran_lambdas[t])
                assert not np.any(np.isnan(Thetas[t])) and not np.any(np.isinf(Thetas[t])), (t, Thetas[t], tran_thetas[t], Lambdas[t], invert(Lambdas[t])@Thetas[t])
                if np.all(obs[t] != np.nan):
                    Thetas[t] = Thetas[t] + Cs[t-1].T @ R_invs[t-1] @ obs[t-1] + emi_thetas[t-1]
            else:
                Lambdas[t] = Lambda_1
                Thetas[t] = Theta_1
                assert not np.any(np.isnan(Lambdas[t])) and not np.any(np.isinf(Lambdas[t])), (t, Lambdas[t], tran_lambdas[t])
                assert not np.any(np.isnan(Thetas[t])) and not np.any(np.isinf(Thetas[t])), (t, Thetas[t], tran_thetas[t])
            Lambdas[t] = turn_matrix_positive_semidefinite(Lambdas[t])
        return Lambdas, Thetas, dyn_thetas

    def dump(self, name, location="./"):
        self.initial.dump(location, name)
        self.transition.dump(location, name)
        self.emission.dump(location, name)
        self.dynamics.dump(location, name)
        self.duration.dump(location, name)

    # @profile
    def _forward_kalman_sample(self, Lambdas, Thetas, states, dyn_thetas):
        """Forward Kalman sampling with optimized Cholesky-based MVN sampling.

        Uses Numba-accelerated Cholesky decomposition when available, falling back
        to scipy.stats.multivariate_normal for numerical stability.
        """
        assert isinstance(states, np.ndarray)
        assert isinstance(Lambdas, np.ndarray)
        assert isinstance(Thetas, np.ndarray)
        states = states.astype(int)
        T = states.shape[0]
        T_ = T if not self.extended_kalman else T + 1
        As = self.dynamics.get_transitions()
        Sigmas = self.dynamics.get_sigmas()
        # Pre-compute all Sigma inverses (one per state)
        Sigma_invs = np.linalg.inv(Sigmas)
        D = As.shape[1]
        res = np.zeros((T_, D))
        x = self.dynamics.sample_obs(0, None)

        # Helper function for optimized MVN sampling
        def _sample_mvn_optimized(mu, S):
            """Sample from MVN using Cholesky when possible, scipy fallback otherwise."""
            if NUMBA_AVAILABLE:
                try:
                    L = cholesky_lower(S)
                    z = np.random.randn(D)
                    return cholesky_mvn_sample(mu, L, z)
                except:
                    pass
            # Fallback to scipy with PSD fixes
            try:
                return multivariate_normal.rvs(mean=mu, cov=S)
            except:
                S_fixed = turn_matrix_positive_semidefinite(S)
                try:
                    return multivariate_normal.rvs(mean=mu, cov=S_fixed)
                except:
                    S_fixed = aggresively_turn_matrix_positive_semidefinite(S)
                    return multivariate_normal.rvs(mean=mu, cov=S_fixed)

        if self.extended_kalman:
            S = invert(Lambdas[0])
            x = _sample_mvn_optimized(S @ Thetas[0], S)
            res[0] = x

        for t in range(T):
            t_ = t if not self.extended_kalman else t + 1
            state = states[t]
            Sigma_inv = Sigma_invs[state]
            assert not np.any(np.isnan(Sigma_inv)), f"Sigma_inv: {Sigma_inv}"

            if t_ > 0:
                S = invert(Sigma_inv + Lambdas[t_])
                mu_new = S @ (Sigma_inv @ (As[state] @ x - dyn_thetas[t]) + Thetas[t_])
            else:
                S = invert(Lambdas[t_])
                mu_new = S @ Thetas[t_]
            assert not np.any(np.isnan(S)), f"S: {S}"

            x = _sample_mvn_optimized(mu_new, S)
            res[t_] = x

        return res

    def _get_likelihood(self, p, x, i, j, di, dj):
        return np.exp(self._get_loglikelihood(p, x, i, j, di, dj))

    def _get_loglikelihood(self, p, x, i, j, di, dj, omega_t=None, omega_d=None):
        if di == 1:
            if isinstance(self.transition, RecurrentTransition):
                res = self.transition.loglikelihood(x, i, j, omega=omega_t)
            else:
                res = self.transition.loglikelihood(i, j)
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                res += self.duration.loglikelihood(j, dj, x, omega=omega_d)
            else:
                res += self.duration.loglikelihood(j, dj)
                # raise Exception("Unknown duration")
            return res
        elif (i == j) and (dj == di - 1):
            return 0
        return -1000000000000

    def loglikelihood(self, Zs, Xs, Ps, Ys, Omegas_T=None, Omegas_D=None):  # TODO przeorać tę funkcję
        """
        Calculates the log likelihood of the model given state
        and observation sequences

        Parameters
        ----------
        ----------
        Zs : list
            list of state sequences
        Ys : list
            list of observation sequences
        """
        loopy = isinstance(self.transition, LoopyTransition) or (
                isinstance(self.transition, RecurrentTransition) and self.transition.loopy)
        l = 0
        if Omegas_T is None:
            Omegas_T = [None] * len(Zs)
        if Omegas_D is None:
            Omegas_D = [None] * len(Zs)
        for Z, X, P, Y, Omegas_t, Omegas_d in zip(Zs, Xs, Ps, Ys, Omegas_T, Omegas_D):
            assert len(X) > len(Omegas_t)
            l += self.initial.loglikelihood(z=Z[0], p=P[0], x=X[0])
            for t in range(1, len(Z)):
                i = Z[t - 1][0]
                j = Z[t][0]
                di = Z[t - 1][1]
                y = Y[t]
                x = X[t + 1]
                p = P[t]
                prev_x = X[t]
                omega_d = omega_t = None
                if Omegas_t is not None:
                    omega_t = Omegas_t[t - 1]
                if Omegas_d is not None:
                    omega_d = Omegas_d[t]
                if (i == j) and not loopy:
                    l += self.emission.loglikelihood(j, x, y)
                    l += self.dynamics.loglikelihood(j, prev_x, x)
                else:
                    if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                        l += self.duration.loglikelihood(i, di, prev_x, omega=omega_d)
                    else:
                        l += self.duration.loglikelihood(i, di)
                        # raise Exception("Unknown duration")
                    if isinstance(self.transition, RecurrentTransition):
                        l += self.transition.loglikelihood(prev_x, i, j, omega=omega_t)
                    else:
                        l += self.transition.loglikelihood(i, j)
                    l += (
                            self.emission.loglikelihood(j, x, y) +
                            self.dynamics.loglikelihood(j, prev_x, x))
        return l

    def _initial_augmented_log_probs(self, X_priors, P_priors, P, X):
        """Return the initial joint log law over ``(duration, state)``."""
        _, right = self.duration.max_support()
        initial_log_probs = np.full((right, self.K), -np.inf)
        for state in self.states:
            P0 = np.minimum(
                np.maximum(
                    P_priors[state] if not self.extended_kalman else P[0],
                    0.00001,
                ),
                0.99999,
            )
            X0 = X_priors[state] if not self.extended_kalman else X[0]
            for duration_index in range(right):
                initial_pair = (state, duration_index + 1)
                if isinstance(self.duration, RecurrentDuration) or isinstance(
                    self.duration, DummyDuration
                ):
                    score = self.initial.loglikelihood(initial_pair, x=X0)
                elif isinstance(self.duration, NonRecurrentDuration):
                    score = self.initial.loglikelihood(initial_pair)
                else:
                    raise ValueError(
                        f"unsupported duration type: {type(self.duration).__name__}"
                    )
                initial_log_probs[duration_index, state] = score
        return initial_log_probs

    def forward_v2(self, X_priors, P_priors, P, X, Y, Omegas_T=None, Omegas_D=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """
        # Validate Omegas_D shape for RecurrentDuration
        if isinstance(self.duration, RecurrentDuration) and Omegas_D is not None:
            assert len(Omegas_D.shape) == 3, \
                f"Omegas_D must be 3D (T, K, D_max-1) for RecurrentDuration, got shape {Omegas_D.shape}"
            if not self.duration.nonswitching:
                assert Omegas_D.shape[1] == self.K, \
                    f"Omegas_D.shape[1]={Omegas_D.shape[1]} must equal K={self.K} for switching RecurrentDuration"
            assert Omegas_D.shape[2] == self.duration.D_max - 1, \
                f"Omegas_D.shape[2]={Omegas_D.shape[2]} must equal D_max-1={self.duration.D_max - 1}"

        log.info('running forward algorithm')
        # initialise alphahat
        limit = len(X) - 1 if self.extended_kalman else len(X)
        left, right = self.duration.max_support()
        alphahat = np.log(np.zeros((limit, right, self.K)))
        switch_lls = np.zeros((limit, self.K, self.K, right))
        log.debug('calculating observation loglikelihoods')
        ol = np.zeros((len(X), right, self.K))
        dl = np.zeros((len(X), right, self.K))
        for i in self.states:
            if not self.extended_kalman:
                prev_x = X_priors[i]
            else:
                prev_x = X[0]
            for t, y in enumerate(Y):
                t_ = t if not self.extended_kalman else t + 1
                x = X[t_]
                dl[t_, :, i] = self.dynamics.loglikelihood(state=i, observation=x,
                                                           previous_observation=prev_x)
                ol[t_, :, i] = dl[t_, :, i] + self.emission.loglikelihood(state=i, observation=y, dynamics=x)
                prev_x = x
        # ol = np.maximum(-1000000000000 * np.ones_like(ol), ol)
        log.debug('starting iteration')

        for t, y in enumerate(Y):
            t_ = t if not self.extended_kalman else t + 1
            x = X[t_]
            if t == 0:
                for i in self.states:
                    P0 = np.minimum(np.maximum(P_priors[i] if not self.extended_kalman else P[0], 0.00001), 0.99999)
                    X0 = X_priors[i] if not self.extended_kalman else X[0]
                    for d in range(right):
                        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                            alphahat[t][d][i] = self.initial.loglikelihood((i, d + 1), x=X0)
                        elif isinstance(self.duration, NonRecurrentDuration):
                            alphahat[t][d][i] = self.initial.loglikelihood((i, d + 1))
                        else:
                            raise Exception("Duration unknown.")
            else:
                tran_ll = np.zeros((self.K, self.K))
                dur_ll = np.zeros((self.K, right))
                for i in self.states:
                    if isinstance(self.transition, RecurrentTransition):
                        assert len(X) >= len(Omegas_T)
                        tran_ll[i] = self.transition.get_log_transition(i, X[t_ - 1], omega=Omegas_T[t - 1])
                    else:
                        tran_ll[i] = self.transition.loglikelihood(i, self.states)
                    if isinstance(self.duration, RecurrentDuration):
                        dur_ll[i] = self.duration.get_log_transition(i, X[t_ - 1], omega=Omegas_D[t_ - 1])
                    else:
                        dur_ll[i] = self.duration.loglikelihood(i, np.arange(right) + 1)
                assert not np.any(np.isnan(alphahat)), alphahat
                assert not np.any(np.isnan(dur_ll)), dur_ll
                assert not np.any(np.isnan(tran_ll)), tran_ll
                log_probs = tran_ll[:, :, np.newaxis] + dur_ll[np.newaxis, :, :]  # TODO einsum
                alphahat[t, :-1, :] = _logsumexp(alphahat[t, :-1, :], alphahat[t - 1, 1:, :])
                for i in self.states:
                    log_prob = log_probs[:, i]
                    alphahat[t, :, i] = _logsumexp(alphahat[t, :, i],
                                                   logsumexp(alphahat[t - 1, 0, :].reshape((1, self.K)) + log_prob.T,
                                                             axis=1))
                switch_lls[t - 1] = log_prob + dl[t_].T[np.newaxis]
            assert not np.any(np.isnan(ol[t_])), ol[t_]
            alphahat[t] = alphahat[t] + ol[t_]
            assert not np.any(np.isnan(ol[t_])), ol[t_]
            assert not np.any(np.isnan(alphahat[t])), alphahat[t]
        return alphahat, switch_lls

    def backward_v2(self, X_priors, P_priors, P, X, Y, Omegas_T=None, Omegas_D=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """
        # Validate Omegas_D shape for RecurrentDuration
        if isinstance(self.duration, RecurrentDuration) and Omegas_D is not None:
            assert len(Omegas_D.shape) == 3, \
                f"Omegas_D must be 3D (T, K, D_max-1) for RecurrentDuration, got shape {Omegas_D.shape}"
            if not self.duration.nonswitching:
                assert Omegas_D.shape[1] == self.K, \
                    f"Omegas_D.shape[1]={Omegas_D.shape[1]} must equal K={self.K} for switching RecurrentDuration"
            assert Omegas_D.shape[2] == self.duration.D_max - 1, \
                f"Omegas_D.shape[2]={Omegas_D.shape[2]} must equal D_max-1={self.duration.D_max - 1}"

        log.info('running forward algorithm')
        # initialise alphahat
        limit = len(X) - 1 if self.extended_kalman else len(X)
        left, right = self.duration.max_support()
        betahat = np.full((limit, right, self.K), -np.inf)
        switch_lls = np.zeros((limit, self.K, self.K, right))
        log.debug('calculating observation loglikelihoods')
        ol = np.zeros((len(X), right, self.K))
        for i in self.states:
            if not self.extended_kalman:
                prev_x = X_priors[i]
            else:
                prev_x = X[0]
            for t, y in enumerate(Y):
                t_ = t if not self.extended_kalman else t + 1
                x = X[t_]
                ol[t_, :, i] = self.dynamics.loglikelihood(state=i, observation=x,
                                                           previous_observation=prev_x) + self.emission.loglikelihood(
                    state=i, observation=y, dynamics=x)
                prev_x = x
        # ol = np.maximum(-1000000000000 * np.ones_like(ol), ol)
        log.debug('starting iteration')
        for t, y in reversed(list(enumerate(Y))):
            t_ = t if not self.extended_kalman else t + 1
            if t == limit - 1:
                betahat[limit - 1] = np.zeros_like(betahat[limit - 1])
            else:
                tran_ll = np.zeros((self.K, self.K))
                dur_ll = np.zeros((self.K, right))
                for i in self.states:
                    if isinstance(self.transition, RecurrentTransition):
                        assert len(X) >= len(Omegas_T)
                        tran_ll[i] = self.transition.get_log_transition(i, X[t_], omega=Omegas_T[t])
                    else:
                        tran_ll[i] = self.transition.loglikelihood(i, self.states)
                    if isinstance(self.duration, RecurrentDuration):
                        dur_ll[i] = self.duration.get_log_transition(i, X[t_], omega=Omegas_D[t_])
                    else:
                        dur_ll[i] = self.duration.loglikelihood(i, np.arange(right) + 1)
                betahat[t, 1:, :] = _logsumexp(betahat[t, 1:, :], betahat[t + 1, :-1, :])
                assert not np.any(np.isnan(betahat)), betahat
                assert not np.any(np.isnan(dur_ll)), dur_ll
                assert not np.any(np.isnan(tran_ll)), tran_ll
                for i in self.states:
                    log_prob = tran_ll[:, i].reshape((self.K, 1)) + dur_ll[np.newaxis, i]  # TODO einsum
                    betahat[t, 0, :] = _logsumexp(logsumexp(betahat[t + 1, :, i, np.newaxis] + log_prob.T,
                                                            axis=0).reshape((1, self.K)), betahat[t, 0, :].reshape((1, self.K)))
                    switch_lls[t, :, i] = log_prob
            betahat[t] += ol[t_]
            # betahat[t] -= logsumexp(betahat[t])
        return betahat, switch_lls

    def beam_forward_v2(self, X_priors, P_priors, P, X, Y, U, W=None, decay=None, Omegas_T=None, Omegas_D=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """
        # Validate Omegas_D shape for RecurrentDuration
        if isinstance(self.duration, RecurrentDuration) and Omegas_D is not None:
            assert len(Omegas_D.shape) == 3, \
                f"Omegas_D must be 3D (T, K, D_max-1) for RecurrentDuration, got shape {Omegas_D.shape}"
            if not self.duration.nonswitching:
                assert Omegas_D.shape[1] == self.K, \
                    f"Omegas_D.shape[1]={Omegas_D.shape[1]} must equal K={self.K} for switching RecurrentDuration"
            assert Omegas_D.shape[2] == self.duration.D_max - 1, \
                f"Omegas_D.shape[2]={Omegas_D.shape[2]} must equal D_max-1={self.duration.D_max - 1}"

        log.info('running forward algorithm')
        if self.nobeam:
            U = np.zeros_like(U)
        # initialise alphahat
        limit = len(X) - 1 if self.extended_kalman else len(X)
        left, right = self.duration.max_support()
        alphahat = np.ones((limit, right, self.K)) * np.log(0.)
        switch_lls = np.zeros((limit, self.K, self.K, right))
        log.debug('calculating observation loglikelihoods')
        ol = np.zeros((len(X), right, self.K))
        dl = np.zeros((len(X), right, self.K))

        # Use batch methods if available for significant speedup
        T_obs = len(Y)
        if hasattr(self.dynamics, 'loglikelihood_batch') and hasattr(self.emission, 'loglikelihood_batch'):
            # Prepare previous observations array
            if not self.extended_kalman:
                # First previous observation comes from X_priors (state-dependent)
                # This requires per-state handling for the first step
                prev_X = np.zeros((T_obs, X.shape[1]))
                prev_X[1:] = X[:-1] if not self.extended_kalman else X[1:-1]
                # Handle first timestep separately
                observations = X if not self.extended_kalman else X[1:]
            else:
                prev_X = X[:-1]
                observations = X[1:]

            # Batch compute dynamics log-likelihoods: returns (K, T)
            dl_batch = self.dynamics.loglikelihood_batch(prev_X, observations)  # (K, T)

            # Batch compute emission log-likelihoods: returns (K, T)
            Y_array = np.array(Y)
            el_batch = self.emission.loglikelihood_batch(observations, Y_array)  # (K, T)

            # Combined log-likelihood: (K, T) -> need to transpose and broadcast to (T, right, K)
            ol_combined = (dl_batch + el_batch).T  # (T, K)

            # Broadcast to duration dimension (log-likelihood same for all durations)
            if not self.extended_kalman:
                ol[:, :, :] = ol_combined[:, np.newaxis, :]
                dl[:, :, :] = dl_batch.T[:, np.newaxis, :]
            else:
                ol[1:, :, :] = ol_combined[:, np.newaxis, :]
                dl[1:, :, :] = dl_batch.T[:, np.newaxis, :]

            # Handle first timestep with state-dependent priors if not extended_kalman
            if not self.extended_kalman:
                for i in self.states:
                    prev_x = X_priors[i]
                    x = X[0]
                    y = Y[0]
                    dl[0, :, i] = self.dynamics.loglikelihood(state=i, observation=x,
                                                               previous_observation=prev_x)
                    ol[0, :, i] = dl[0, :, i] + self.emission.loglikelihood(state=i, observation=y, dynamics=x)
        else:
            # Fallback to original loop
            for i in self.states:
                if not self.extended_kalman:
                    prev_x = X_priors[i]
                else:
                    prev_x = X[0]
                for t, y in enumerate(Y):
                    t_ = t if not self.extended_kalman else t + 1
                    x = X[t_]
                    dl[t_, :, i] = self.dynamics.loglikelihood(state=i, observation=x,
                                                               previous_observation=prev_x)
                    ol[t_, :, i] = dl[t_, :, i] + self.emission.loglikelihood(state=i, observation=y, dynamics=x)
                    prev_x = x
        log.debug('starting iteration')

        for t, y in enumerate(Y):
            t_ = t if not self.extended_kalman else t + 1
            x = X[t_]
            if t == 0:
                for i in self.states:
                    P0 = np.minimum(np.maximum(P_priors[i] if not self.extended_kalman else P[0], 0.00001), 0.99999)
                    X0 = X_priors[i] if not self.extended_kalman else X[0]
                    for d in range(right):
                        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                            alphahat[t][d][i] = self.initial.loglikelihood((i, d + 1), x=X0)
                        else:
                            alphahat[t][d][i] = self.initial.loglikelihood((i, d + 1))
                            # raise Exception("Duration unknown.")
            else:
                # Optimized: compute all transition log-likelihoods at once
                if isinstance(self.transition, RecurrentTransition):
                    # Use batch method: returns (1, K, K) then squeeze to (K, K)
                    tran_ll = self.transition.get_log_transition_batch(
                        X[t_ - 1:t_], inputs=None
                    )[0]  # Single timestep, get first element
                else:
                    tran_ll = np.zeros((self.K, self.K))
                    for i in self.states:
                        tran_ll[i] = self.transition.loglikelihood(i, self.states)

                # Duration log-likelihoods (still per-state for now)
                dur_ll = np.zeros((self.K, right))
                for i in self.states:
                    if isinstance(self.duration, RecurrentDuration):
                        dur_ll[i] = self.duration.get_log_transition(i, X[t_ - 1], omega=Omegas_D[t_ - 1])
                    else:
                        dur_ll[i] = self.duration.loglikelihood(i, np.arange(right) + 1)
                u = U[t]
                log_probs = tran_ll[:, :, np.newaxis] + dur_ll[np.newaxis, :, :]  # TODO einsum
                alphahat[t, :-1, :] = _logsumexp(alphahat[t, :-1, :], alphahat[t - 1, 1:, :])
                for i in self.states:
                    log_prob = log_probs[:, i]
                    prob = np.exp(log_prob)  # TODO einsum
                    reachable = prob.T > u
                    log_mask = np.log(reachable.astype(float))
                    switch_lls[t - 1, :, i] = log_mask.T
                    alphahat[t, :, i] = _logsumexp(alphahat[t, :, i],
                                                   logsumexp(alphahat[t - 1, 0, :] + log_mask, axis=1))
                # switch_lls[t-1] += ol[t_].T[np.newaxis]
            assert not np.any(np.isnan(ol[t_])), ol[t_]
            alphahat[t] = alphahat[t] + ol[t_]
            assert not np.any(np.isnan(ol[t_])), ol[t_]
            assert not np.any(np.isnan(alphahat[t])), alphahat[t]
        return alphahat, switch_lls

    def _random_state_init(self, L):
        res_z = np.zeros(L)
        res_d = np.zeros(L)
        z, d = self.initial.sample()[:2]  # LearnableInitial also returns p / x0
        for i in range(L):
            res_z[i] = z
            res_d[i] = d
            if d > 1:
                d -= 1
            else:
                x = self.dynamics.sample_obs(z)
                if isinstance(self.transition, RecurrentTransition):
                    z = self.transition.sample_x(z, x)
                else:
                    z = self.transition.sample_x(z)
                if isinstance(self.duration, RecurrentDuration):
                    d = self.duration.sample_d(z, x)
                elif isinstance(self.duration, NonRecurrentDuration):
                    d = self.duration.sample_d(z)
                else:
                    raise Exception("Unknown duration")
        return res_z.astype(int), res_d.astype(int)

    def slice_sample(self, Z, P, X, min_u=0):
        log.info('forming slice')
        if self.nobeam:
            return np.zeros(len(Z))
        u = [np.array(min_u)]
        for t in range(1, len(Z)):
            t_ = t if not self.extended_kalman else t + 1
            i = Z[t - 1][0]
            j = Z[t][0]
            di = Z[t - 1][1]
            dj = Z[t][1]
            try:
                u.append(
                    np.random.uniform(
                        low=min_u,
                        high=self._get_likelihood(i=i, j=j, di=di, dj=dj, x=X[t_ - 1], p=P[t_ - 1]))
                )
            except KeyError:
                raise
        return np.array(u)

    def forward_sample_v2(self, betahat, switch_lls, *, initial_log_probs):
        T = len(betahat)

        def require_finite_path(scores, context):
            if not np.isfinite(logsumexp(scores)):
                raise ValueError(
                    f"forward discrete sampler has no finite {context} path"
                )
            return scores

        initial_scores = betahat[0] + initial_log_probs
        require_finite_path(initial_scores, "initial")
        Z = [self._sample_z(initial_scores.T, np.ones_like(initial_scores.T).astype(bool))]
        for edge_t in range(T - 1):
            s, d = Z[-1]
            if d > 1:
                Z.append((s, d - 1))
                continue
            a = betahat[edge_t + 1] + switch_lls[edge_t, s].T
            require_finite_path(a, f"switch at edge {edge_t}")
            z = self._sample_z(a.T, np.ones_like(a).T.astype(bool))
            Z.append(z)
        return Z

    def backward_sample_v2(self, alphahat, switch_lls):
        T = len(alphahat)
        try:
            Z = [self._sample_z(alphahat[-1].T, np.ones_like(alphahat[-1].T).astype(bool))]
        except ValueError:
            print(alphahat[-1])
            raise
        ones_mask = np.zeros_like(alphahat[0]).astype(bool)
        ones_mask[0, :] = True
        for t in reversed(range(T - 1)):
            s, d = Z[-1]
            mask = ones_mask.copy()
            if d < mask.shape[0]:
                mask[d, s] = True
            a = alphahat[t]
            a[0, :] = a[0, :] + switch_lls[t, :, s, d - 1]
            z = self._sample_z(a.T, mask.T)
            Z.append(z)
        Z.reverse()
        return Z

    def _sample_z(self, alphahat, reachable):
        candidates = np.stack(np.nonzero(reachable), axis=1)
        lls = alphahat[reachable]
        m = lls.max()
        probs = np.exp(lls - m)
        try:
            xi = Categorical(probs).sample()
        except Exception:
            probs = np.ones_like(probs)
            probs /= probs.sum()
            xi = Categorical(probs).sample()
        return candidates[xi] + np.arange(2)

    def _beam(self, name, online, count, sample_U, Z_samples, min_u, log_to_wandb, P, epsilons, X, Y, Omegas_D,
              Omegas_T, decay, plot, plot_folder, burnin, dump_period, dump_path, num_of_workers, fast=False,
              rotate=False,
              normalize_emission=False, freeze_dynamics=False, freeze_transition=False, freeze_duration=False,
              freeze_emission=False, save_space=True, skip_duration_message=False, skip_transition_message=False):
        log.info('\n\nrunning sample %s' % count)
        skip_duration_message = skip_duration_message and (count <= 1500)
        if self.nobeam:
            sample_U = False
            min_u = 0.
        assert len(Omegas_D[0]) == len(Omegas_T[0]), (Omegas_D[0].shape, Omegas_T[0].shape)
        # slice
        start = time.time()
        X_priors = [self.dynamics.sample_obs(k) for k in self.states]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, NonRecurrentDuration):
            P_priors = X_priors
        else:
            raise Exception("Unknown duration")

        if sample_U:
            U = []
            for (Z, X_, P_) in zip(Z_samples, X, P):
                U.append(self.slice_sample(Z, X=X_, P=P_, min_u=min_u))
        else:
            U = [np.zeros(len(Z)) for Z in Z_samples]
        slice_seconds = time.time() - start
        log.debug('slice sample took %ss' % slice_seconds)

        # states
        start = time.time()
        d_stats, d_max_t = None, None
        if online:
            with Pool(num_of_workers, len(X)) as pool:
                if self.forward_discrete_states_sample:
                    initial_log_probs = [
                        self._initial_augmented_log_probs(
                            X_priors, P_priors, Pi, Xi
                        )
                        for Pi, Xi in zip(P, X)
                    ]
                    betas = pool.map(
                        self._betas_generator(P, X, Y, U, X_priors=X_priors, P_priors=P_priors, Omegas_T=Omegas_T,
                                              Omegas_D=Omegas_D, decay=decay),
                        range(len(X)))
                    Z_samples = pool.map(
                        self._Zs_generator(
                            betas, initial_log_probs=initial_log_probs
                        ),
                        range(len(X)))
                else:
                    alphas = pool.map(
                        self._alphas_generator(P, X, Y, U, X_priors=X_priors, P_priors=P_priors, Omegas_T=Omegas_T,
                                               Omegas_D=Omegas_D, decay=decay),
                        range(len(X)))
                    Z_samples = pool.map(
                        self._Zs_generator(alphas),
                        range(len(X)))
                    if not self.nobeam:
                        d_right = self.duration.max_support()[1]
                        d_max_t, d_card_t = beam_truncation_trace([a[1] for a in alphas], d_right)
                        d_stats = beam_truncation_stats(d_max_t, d_card_t, d_right)
            forward_message_seconds = time.time() - start
            log.debug('inference took %ss' % forward_message_seconds)
        else:
            raise NotImplementedError
        Z_samples_expanded = self._expand_Z(Z_samples)
        D_samples_expanded = self._expand_D(Z_samples)
        Y_est = None
        if plot:
            Y_est = self.emission.generate_centers(Z_samples_expanded, X)
            self._plot_everything(plot_folder, name, count, X[0], np.array(Y[0]), np.array(Y_est[0]),
                                  Z_samples_expanded[0], self.dynamics.As, D_samples_expanded[0], len(self.states),
                                  log_to_wandb, save_space=save_space)
        prev_X = X
        prev_Y_est = Y_est
        if self.extended_kalman:
            X_long = X
            X_short = [x[1:] for x in X]
        else:
            # X_priors = [self.dynamics.sample_obs(k) for k in self.states]
            X_long = [self._prepend_initial_latent(X_priors[0], x) for x in X]
            X_short = X
        Omegas_T = [self.transition.sample_Omegas(Z_, X_) for Z_, X_ in zip(Z_samples_expanded, X_short)]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration) or isinstance(
                self.duration, NonRecurrentDuration):
            Omegas_D = [self.duration.sample_Omegas(Z_, X_, D_) for Z_, X_, D_ in
                        zip(Z_samples_expanded, X_long, D_samples_expanded)]
        else:
            raise Exception("Unknown duration")
        start = time.time()
        with Pool(num_of_workers, len(Y)) as pool:
            assert not np.any(np.isnan(np.concatenate(Omegas_T))), f"Omegas_T: {Omegas_T}"
            assert not np.any(np.isnan(np.concatenate(Omegas_D))), f"Omegas_D: {Omegas_D}"
            X = pool.map(
                self._kalman_generator(Y, Z_samples_expanded, D_samples_expanded, epsilons=epsilons, Omegas_D=Omegas_D,
                                       Omegas_T=Omegas_T,
                                       skip_duration_message=skip_duration_message,
                                       skip_transition_message=skip_transition_message),
                range(len(Y)))
            self._check_latent(X)   # before the parameter updates and the dump consume a diverged trajectory
        kalman_seconds = time.time() - start
        if not freeze_emission:
            self.emission.update(Zs=Z_samples_expanded, Xs=X_short, Ys=Y, normalize=normalize_emission)
            if rotate:
                rotation = self.emission.rotate()
                self.dynamics.rotate(rotation)
                X = [self.dynamics.rotate_samples(rotation, X_) for X_ in X]
                if self.extended_kalman:
                    X_long = X
                    X_short = [x[1:] for x in X]
                else:
                    X_long = [self._prepend_initial_latent(X_priors[0], x) for x in X]
                    X_short = X
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, NonRecurrentDuration):
            Psis = None
            epsilons = [np.zeros(Xi.shape[0]) for Xi in X]
            P = X.copy()
        else:
            raise Exception("Unknown duration")
        switches = [np.ones_like(d).astype(bool) for d in D_samples_expanded]
        if not self.ignore_switches:
            for i, d in enumerate(D_samples_expanded):
                switches[i][1:] = (d == 1)[:-1]
        if not freeze_duration:
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                self.duration.update(Z=Z_samples_expanded, X=X_long, D=D_samples_expanded, Omegas=Omegas_D,
                                     switches=switches)
            elif isinstance(self.duration, NonRecurrentDuration):
                self.duration.update(Z_samples)
            else:
                raise Exception("Unknown duration")
        if not freeze_transition:
            if isinstance(self.transition, RecurrentTransition):
                self.transition.update(Z_samples_expanded, X_short, Omegas_T, switches=switches)
            else:
                self.transition.update(Z_samples)
        if not freeze_dynamics:
            self.dynamics.update(Xs=X_long, Zs=Z_samples_expanded)
        l = self.loglikelihood(Zs=Z_samples, Xs=X, Ys=Y, Ps=P, Omegas_T=Omegas_T, Omegas_D=Omegas_D)
        if log_to_wandb:
            import wandb
            self.duration.log_to_wandb(count)
            self.transition.log_to_wandb(count)
            self.dynamics.log_to_wandb(count)
            self.emission.log_to_wandb(count)
            wandb.log({"loglikelihood": l}, step=count)
        log.info("log loglikelihood at iteration %s: %s" % (count, l))

        if count > burnin:
            if count % dump_period == 0:
                log.debug('writing iteration %s to disk' % count)
                self.dump(name, dump_path)
        return {"prev_X": prev_X,
                "X": X,
                "X_short": X_short,
                "Z_samples": Z_samples,
                "prev_Y_est": prev_Y_est,
                "P": P,
                "Omegas_T": Omegas_T,
                "Omegas_D": Omegas_D,
                "loglikelihood": l,
                "sweep_stats": {"slice_seconds": slice_seconds,
                                "forward_message_seconds": forward_message_seconds,
                                "kalman_seconds": kalman_seconds,
                                "d_stats": d_stats,
                                **({} if freeze_dynamics else self.dynamics.draw_stats)},
                "d_max_t": d_max_t,
                }

    def _infer(self, online, count, sample_U, Z_samples, min_u, P, epsilons, X, Y, Omegas_D,
               Omegas_T, decay, num_of_workers, fast=False, skip_duration_message=False, skip_transition_message=False):
        log.info('\n\nrunning sample %s' % count)
        if self.nobeam:
            sample_U = False
            min_u = 0.
        assert len(Omegas_D[0]) == len(Omegas_T[0]), (Omegas_D[0].shape, Omegas_T[0].shape)
        # slice
        start = time.time()
        X_priors = [self.dynamics.sample_obs(k) for k in self.states]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, NonRecurrentDuration):
            P_priors = X_priors
        else:
            raise Exception("Unknown duration")

        if sample_U:
            U = []
            for (Z, X_, P_) in zip(Z_samples, X, P):
                U.append(self.slice_sample(Z, X=X_, P=P_, min_u=min_u))
        else:
            U = [np.zeros(len(Z)) for Z in Z_samples]
        log.debug('slice sample took %ss' % (time.time() - start))

        # states
        start = time.time()
        if online:
            with Pool(num_of_workers, len(X)) as pool:
                if self.forward_discrete_states_sample:
                    initial_log_probs = [
                        self._initial_augmented_log_probs(
                            X_priors, P_priors, Pi, Xi
                        )
                        for Pi, Xi in zip(P, X)
                    ]
                    betas = pool.map(
                        self._betas_generator(P, X, Y, U, X_priors=X_priors, P_priors=P_priors,
                                              Omegas_T=Omegas_T, Omegas_D=Omegas_D, decay=decay),
                        range(len(X)))
                    Z_samples = pool.map(
                        self._Zs_generator(
                            betas, initial_log_probs=initial_log_probs
                        ),
                        range(len(X)))
                else:
                    alphas = pool.map(
                        self._alphas_generator(P, X, Y, U, X_priors=X_priors, P_priors=P_priors,
                                               Omegas_T=Omegas_T, Omegas_D=Omegas_D, decay=decay),
                        range(len(X)))
                    Z_samples = pool.map(
                        self._Zs_generator(alphas),
                        range(len(X)))
            log.debug('inference took %ss' % (time.time() - start))
        else:
            raise NotImplementedError
        Z_samples_expanded = self._expand_Z(Z_samples)
        D_samples_expanded = self._expand_D(Z_samples)
        if self.extended_kalman:
            X_long = X
            X_short = [x[1:] for x in X]
        else:
            X_long = [self._prepend_initial_latent(X_priors[0], x) for x in X]
            X_short = X
        Omegas_T = [self.transition.sample_Omegas(Z_, X_) for Z_, X_ in zip(Z_samples_expanded, X_short)]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration) or isinstance(
                self.duration, NonRecurrentDuration):
            Omegas_D = [self.duration.sample_Omegas(Z_, X_, D_) for Z_, X_, D_ in
                        zip(Z_samples_expanded, X_long, D_samples_expanded)]
        else:
            raise Exception("Unknown duration")
        with Pool(num_of_workers, len(Y)) as pool:
            assert not np.any(np.isnan(np.concatenate(Omegas_T))), f"Omegas_T: {Omegas_T}"
            assert not np.any(np.isnan(np.concatenate(Omegas_D))), f"Omegas_D: {Omegas_D}"
            X = pool.map(
                self._kalman_generator(Y, Z_samples_expanded, D_samples_expanded, epsilons=epsilons, Omegas_D=Omegas_D,
                                       Omegas_T=Omegas_T,
                                       skip_duration_message=skip_duration_message,
                                       skip_transition_message=skip_transition_message),
                range(len(Y)))
            assert not np.any(np.isnan(np.concatenate(X))), f"X: {X}"
        if self.extended_kalman:
            X_long = X
            X_short = [x[1:] for x in X]
        else:
            X_long = [self._prepend_initial_latent(X_priors[0], x) for x in X]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, NonRecurrentDuration):
            Psis = None
            epsilons = [np.zeros(Xi.shape[0]) for Xi in X]
            P = X.copy()
        else:
            raise Exception("Unknown duration")
        l = self.loglikelihood(Zs=Z_samples, Xs=X, Ys=Y, Ps=P, Omegas_T=Omegas_T, Omegas_D=Omegas_D)
        return {
            "X": X,
            "X_short": X_short,
            "Z_samples": Z_samples,
            "P": P,
            "Omegas_T": Omegas_T,
            "Omegas_D": Omegas_D,
            "loglikelihood": l
        }

    def _kalman_generator(self, Y, Z, D, Omegas_D, Omegas_T, epsilons, skip_duration_message=False,
                          skip_transition_message=False):
        def _kalman_pass(i):
            y = Y[i]
            z = Z[i]
            d = D[i]
            omegas_d = Omegas_D[i]
            omegas_t = Omegas_T[i]
            epsilonsi = epsilons[i]
            Lambdas, Thetas, dyn_thetas = self._backward_kalman_filter(y, z, d,
                                                                       omegas_dur=omegas_d,
                                                                       epsilons=epsilonsi,
                                                                       omegas_tran=omegas_t,
                                                                       skip_duration_message=skip_duration_message,
                                                                       skip_transition_message=skip_transition_message)
            Xi = self._forward_kalman_sample(Lambdas, Thetas, z, dyn_thetas)
            return Xi

        return _kalman_pass

    def _alphas_generator(self, P, X, Y, U, X_priors, P_priors, Omegas_T, Omegas_D, decay):
        if self.nobeam:
            U = [np.zeros_like(u) for u in U]

        def _alphas_pass(i):
            Xi = X[i]
            Yi = Y[i]
            Pi = P[i]
            Omegas_Ti = Omegas_T[i]
            Omegas_Di = Omegas_D[i]
            if self.nobeam:
                return self.forward_v2(X_priors=X_priors, P_priors=P_priors, P=Pi, X=Xi, Y=Yi, Omegas_T=Omegas_Ti,
                                       Omegas_D=Omegas_Di)
            else:
                return self.beam_forward_v2(X_priors=X_priors, P_priors=P_priors, P=Pi, X=Xi, Y=Yi, U=U[i], decay=decay,
                                            Omegas_T=Omegas_Ti, Omegas_D=Omegas_Di)

        return _alphas_pass

    def _betas_generator(self, P, X, Y, U, X_priors, P_priors, Omegas_T, Omegas_D, decay):
        if self.nobeam:
            U = [np.zeros_like(u) for u in U]

        def _betas_pass(i):
            Xi = X[i]
            Yi = Y[i]
            Pi = P[i]
            Omegas_Ti = Omegas_T[i]
            Omegas_Di = Omegas_D[i]
            if self.nobeam:
                return self.backward_v2(X_priors=X_priors, P_priors=P_priors, P=Pi, X=Xi, Y=Yi, Omegas_T=Omegas_Ti,
                                        Omegas_D=Omegas_Di)
            else:
                raise NotImplementedError

        return _betas_pass

    def _Zs_generator(self, alphas, *, initial_log_probs=None):
        def _alphas_pass(i):
            return self.backward_sample_v2(alphas[i][0], alphas[i][1])

        def _betas_pass(i):
            return self.forward_sample_v2(
                alphas[i][0],
                alphas[i][1],
                initial_log_probs=initial_log_probs[i],
            )

        if self.forward_discrete_states_sample:
            if initial_log_probs is None:
                raise ValueError(
                    "initial_log_probs are required for forward discrete sampling"
                )
            return _betas_pass
        return _alphas_pass

    # Bound on the latent trajectory relative to the initial one. It only has to catch true numerical
    # blow-ups (inf, 1e30): the stability-truncated dynamics draw keeps a large trajectory from poisoning
    # the regression, and healthy chains on long sequences can visit |x| ~ 1e4-1e7.
    X_SCALE_FACTOR = 1e8
    SWEEP_REDRAWS = 2          # redraws per rollback level; the slice sampler makes redraws near-deterministic
    ROLLBACK_DEPTH = 3         # earlier sweeps whose start state can be restored
    SWEEP_INPUTS = ("X", "Z_samples", "P", "Omegas_D", "Omegas_T")   # latent inputs of a sweep (_beam kwargs)
    _COMPONENTS = ("duration", "transition", "dynamics", "emission")

    def _snapshot(self):
        return {c: copy.deepcopy(getattr(self, c)) for c in self._COMPONENTS}

    def _restore(self, snapshot):
        for c, value in snapshot.items():
            setattr(self, c, copy.deepcopy(value))
        for attr, c in (("dur_dist", "duration"), ("dynamics_dist", "dynamics")):
            if hasattr(self.initial, attr):
                setattr(self.initial, attr, getattr(self, c))

    def _divergence_evidence(self):
        """One line on the state that produced a diverging sweep: dynamics spectral radius, duration weights."""
        d = self.dynamics.x_dim
        rho = max(np.abs(np.linalg.eigvals(np.asarray(A)[:d, :d])).max() for A in self.dynamics.As)
        beta = getattr(self.duration, "beta", None)
        return f"spectral radius(A) {rho:.3g}" + (f", max|beta| {np.abs(beta).max():.3g}" if beta is not None else "")

    @staticmethod
    def _max_abs(X):
        return max((float(np.abs(x).max()) for x in X if x.size), default=0.)

    def _set_latent_scale(self, X):
        """Fix the divergence bound from the initial latent trajectory (see X_SCALE_FACTOR)."""
        self._x_abs_max = self.X_SCALE_FACTOR * max(self._max_abs(X), 1.)

    def _check_latent(self, X):
        """Max |x| of the trajectory (the sidecar's ``max_abs_x``); raises past the bound or on inf/nan."""
        worst = self._max_abs(X)
        if not np.isfinite(worst) or worst > self._x_abs_max:
            raise FloatingPointError(f"latent trajectory diverged (max |x| = {worst:.3g})")
        return worst

    def _redraw_on_divergence(self, sweep, count, inputs, history):
        """Run one Gibbs sweep; if its numerics blow up, restore an earlier chain state and redraw.

        ``sweep(inputs)`` runs the sweep from the latent inputs (X, z, P,
        Polya-Gamma draws) it is given; ``history`` holds the start states of
        the last accepted sweeps and is updated in place. With the weak
        mixed-prior dynamics prior (dyn_V_0 = 1e-8) a state can draw a
        slightly explosive A with a tiny noise; a long segment in that state
        then makes the Kalman draw grow geometrically (seen with nobeam and
        non-recurrent durations on long sequences). Such a trajectory poisons the next
        dynamics regression, so the sweep is rejected and redrawn from the
        state it started with; if the redraws fail too, the explosive draw
        sits in that state, and the chain is rolled back one sweep at a time
        (parameters and latent inputs together, up to ROLLBACK_DEPTH sweeps)
        and redrawn from there. Every restored state is an accepted state of
        the chain; the number of rejected draws is recorded per sweep in the
        sidecar (``rejected_sweeps``, with the sweep's ``max_abs_x``).
        """
        candidates = history + [(self._snapshot(), inputs)]
        rejected, error = 0, None
        for level, (params, level_inputs) in enumerate(reversed(candidates)):
            if level:
                log.warning("sweep %s: all redraws diverged, rolling back %d sweep(s)", count, level)
                self._restore(params)
            for attempt in range(self.SWEEP_REDRAWS + 1):
                try:
                    res = sweep(level_inputs)
                    max_abs_x = self._check_latent(res["X"])
                    res["sweep_stats"].update(rejected_sweeps=rejected, max_abs_x=max_abs_x)
                    history[:] = candidates[:len(candidates) - level][-self.ROLLBACK_DEPTH:]
                    return res
                except (AssertionError, FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                    traceback.clear_frames(exc.__traceback__)   # the failed sweep's messages and draws (~70 MB on brain)
                    rejected, error = rejected + 1, exc
                    if attempt == self.SWEEP_REDRAWS:
                        break
                    log.warning("sweep %s redraw %d/%d after %s: %s | %s", count, attempt + 1, self.SWEEP_REDRAWS,
                                type(exc).__name__, str(exc)[:120], self._divergence_evidence())
                    self._restore(params)
        raise error

    def _relabelled_accuracy(self, Z_expanded, actual):
        """Hungarian-align the sampled states to `actual`; return (Z, {accuracy, weighted/macro/micro F1})."""
        Z = permute(np.concatenate(Z_expanded), actual, self.K)  # permute() returns Z unchanged on failure
        scores = {"accuracy": float(accuracy_score(Z, actual))}
        scores.update({f"{avg}_f1": float(f1_score(actual, Z, average=avg)) for avg in ("weighted", "macro", "micro")})
        return Z, scores

    def _expand_Z(self, Z):
        return [np.array([x[0] for x in z]) for z in Z]

    def _expand_D(self, Z):
        return [np.array([x[1] for x in z]) for z in Z]

    def _recount_D(self, Z):
        res = []
        for z in Z:
            L = len(z)
            counts = np.ones(L, dtype=int)
            if not isinstance(self.duration, DummyDuration):
                ls = -1
                ld = 0
                for k in range(1, L + 1):
                    if z[-k][0] == ls:
                        counts[-k] = 1 + ld
                    ls = z[-k][0]
                    ld = counts[-k]
            res.append(counts)
        return res

    def _plot_decision_boundary(self, X_min, X_max, state, title=None, fname=None):
        fig = plt.figure(figsize=(20, 10), dpi=80)
        ax = fig.gca()
        a, b = np.meshgrid(np.linspace(X_min[0] - 0.5, X_max[0] + 0.5, 500),
                           np.linspace(X_min[1] - 0.5, X_max[1] + 0.5, 500))
        ab = np.array([a.flatten(), b.flatten()]).T
        z_pred = np.argmax(self.transition.get_log_transition((state * np.ones(len(ab))).astype(int), ab), axis=1)
        ax.scatter(x=a, y=b, c=z_pred)
        display = DecisionBoundaryDisplay(
            xx0=a, xx1=b, response=z_pred.reshape(a.shape)
        )
        display.plot()
        if title:
            display.ax_.set_title(title)
        if fname:
            plt.savefig(fname)
            if "pdf" not in fname:
                path = fname[:-3] + "pdf"
                plt.savefig(path)
        plt.close()

    # @profile
    def _plot_everything(self, path, name, iter, X, Y, Y_est, Z, As, Dseq, states_n, log_to_wandb, save_space=True):
        if not save_space:
            name = f"{name}_{iter}"
        plot_dict = {}
        if self.extended_kalman and len(X) > len(Z):
            X = X[1:]
        file_path = os.path.join(path, f"{name}_dynamics_actual_shift.png")
        plot_actual_shifts(X, Z, states_n, file_path, title=f"{name} - as - {iter}")
        if log_to_wandb:
            import wandb
            plot_dict["dynamics_actual_shift"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_estimated_dynamics_time.png")
        plot_observations(Z[:1000], X[:1000], fname=file_path, title=f"{name} - estimated_dynamics_time - {iter}")
        if log_to_wandb:
            plot_dict["estimated_dynamics_time"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_phase_portrait.png")
        if self.dynamics.affine:
            plot_phase_portrait(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), Z, As, states_n, file_path,
                                title=f"{name} - pp - {iter}")
        else:
            plot_phase_portrait(X, Z, As, states_n, file_path, title=f"{name} - pp - {iter}")
        if log_to_wandb:
            plot_dict["phase_portrait"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_emissions.png")
        plot_emission(Y, Z, states_n, file_path, title=f"{name} - emission - {iter}")
        if log_to_wandb:
            plot_dict["emissions"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_estimated_emissions.png")
        plot_emission(Y_est, Z, states_n, file_path, title=f"{name} - estimated_emission - {iter}")
        if log_to_wandb:
            plot_dict["estimated_emissions"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_estimated_emissions_time.png")
        plot_observations(Z[:1000], Y_est[:1000], fname=file_path, title=f"{name} - estimated_emission_time - {iter}")
        if log_to_wandb:
            plot_dict["estimated_emissions_time"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_durations.png")
        fig, ax = plot_durations(Z, Dseq, states_n, title=f"{name} - durations - {iter}")
        fig.savefig(file_path)
        plt.close(fig)
        if isinstance(self.transition, RecurrentTransition) and (self.dynamics.x_dim == 2) and self.transition.loopy:
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            for k in range(len(self.transition.R)):
                file_path = os.path.join(path, f"{name}_{k}_decision_boundary.png")
                self._plot_decision_boundary(X_min, X_max, k, title=f"{name} - decision_boundary - {k} - {iter}",
                                             fname=file_path)
                # plt.close(fig)
        if log_to_wandb:
            plot_dict["durations"] = wandb.Image(file_path)
            wandb.log(plot_dict, step=iter)
        plt.close('all')

    def _merge(self, Zs, Ds):
        res = []
        for Z, D in zip(Zs, Ds):
            res.append(list(zip(Z, D)))
        return res

    def infer(self, Y: List[np.ndarray], min_u: float = 0., its: int = 100,
              online: bool = True, sample_U: bool = True,
              force_U: Optional[float] = None, decay: Optional[float] = None,
              num_of_workers: int = 5, fast=True, skip_duration_message=False, skip_transition_message=False,
              burnin: int = 0, diagnostics=None):
        """
        Runs the beam sampling approach for the EDHMM

        Parameters
        ----------
        Y : list
            list of observation sequences

        Optional Parameters
        -------------------
        min_u : scalar (0)
            the minimum auxilliary variable to consider. You can use this to tune
            the algorithm. See the paper for details.
        its : integer (100)
            number of iterations to perform
        burnin : integer (50)
            allow this many iterations before writing samples to disk
        name : string (beamer)
            name of the experiment. This will be prepended to all output files
        online : boolean (True)
            whether or not to run the algo online. Currently this has to be True
        sample_U : boolean (True)
            whether or not to update the auxilliary variable. You probably should leave
            this to True, unless you're poking at the algorithm to see what it does
        updated_D : boolean (True)
            whether or not to update the duration distribution. Again, you should leave
            this to True.
        force_U : None or list of lists (None)
            you can force U to start off from a specific starting place if you like. If
            you set this and set sample_U to False then this auxilliary variable won't
            change throughout the algo.
        min_d : None or list of ints
            You can force a minimum duration per state, if you'd like. You must also set
            max_d if you set min_d.
        max_d : None or list of ints
            maximum duration per state
        """
        self._validate_discrete_sampler_configuration()
        if self.nobeam:
            min_u = 0.
            sample_U = False
            force_U = [np.zeros(len(Yi)) for Yi in Y]
            fast = True
        # get support of duration distributions

        # sample auxillary variables from some small value
        if force_U:
            U = force_U
        else:
            U = [np.random.uniform(min_u, 0.000000000001, size=len(Yi)) for Yi in Y]

        # get worthy samples given the relaxed U
        alphas = []
        betas = []
        X = []
        Z_samples_expanded, D_samples_expanded = zip(*[self._random_state_init(len(y)) for y in Y])
        Omegas_D = [self.duration.init_pg(len(y)) for y in Y]
        Omegas_T = [self.transition.init_pg(len(y)) for y in Y]
        P = []
        epsilons = []
        for i, y in enumerate(Y):
            Xi = np.zeros((y.shape[0], self.dynamics.x_dim))
            X.append(Xi)
            l = len(Z_samples_expanded[i])
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                P.append(Xi[:l])
                epsilons.append(np.zeros(l))
            else:
                P.append(Xi[:l])
                epsilons.append(np.zeros(l))
        X_initialized = len(X) != 0
        for i, y in enumerate(Y):
            if not X_initialized:
                Lambdas, Thetas, dyn_thetas = self._backward_kalman_filter(y,
                                                                           Z_samples_expanded[
                                                                               i],
                                                                           D_samples_expanded[
                                                                               i],
                                                                           omegas_dur=Omegas_D[i],
                                                                           omegas_tran=Omegas_T[i],
                                                                           epsilons=epsilons,
                                                                           skip_duration_message=skip_duration_message,
                                                                           skip_transition_message=skip_transition_message)
                Xi = self._forward_kalman_sample(Lambdas, Thetas, Z_samples_expanded[i], dyn_thetas)
                X.append(Xi)
            else:
                if self.extended_kalman:
                    X[i] = np.insert(X[i], 0, X[i][0], axis=0)
                Xi = X[i]
        P = []
        epsilons = []
        for i, y in enumerate(Y):
            Xi = X[i]
            l = len(Z_samples_expanded[i])
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                P.append(Xi[-l:])
                epsilons.append(np.zeros(l))
            else:
                P.append(Xi[-l:])
                epsilons.append(np.zeros(l))
                # raise Exception("Unknown distribution")
        Z_samples = []
        if online:
            for i, Xi in enumerate(X):
                Yi = Y[i]
                X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                    P_priors = X_priors
                else:
                    P_priors = X_priors
                    # raise Exception("Unknown duration")
                if self.forward_discrete_states_sample and self.nobeam:
                    initial_log_probs = self._initial_augmented_log_probs(
                        X_priors, P_priors, P[i], Xi
                    )
                    betas.append(
                        self.backward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, Y=Yi,
                                         P=P[i], Omegas_T=Omegas_T[i], Omegas_D=Omegas_D[i]))
                    Z_samples.append(
                        self.forward_sample_v2(
                            betas[i][0],
                            betas[i][1],
                            initial_log_probs=initial_log_probs,
                        ))
                elif self.nobeam:
                    alphas.append(
                        self.forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, Y=Yi,
                                        P=P[i], Omegas_T=Omegas_T[i], Omegas_D=Omegas_D[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))
                else:
                    alphas.append(
                        self.beam_forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, Y=Yi, U=U[i], decay=decay,
                                             P=P[i], Omegas_T=Omegas_T[i], Omegas_D=Omegas_D[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))
                    # Z_samples.append(
                    #     self.beam_backward_sample_v2(alphas[i][0], alphas[i][1]))
        else:
            raise NotImplementedError
        # count how many iterations we've done so far
        count = 0
        lower = count + 1
        upper = count + 1 + its
        loglikelihoods = []
        for count in range(lower, upper):
            res = self._infer(online=online, count=count, sample_U=sample_U, Z_samples=Z_samples, min_u=min_u,
                              X=X, Y=Y, decay=decay,
                              num_of_workers=num_of_workers, Omegas_D=Omegas_D, Omegas_T=Omegas_T, P=P,
                              epsilons=epsilons, fast=fast, skip_duration_message=skip_duration_message,
                              skip_transition_message=skip_transition_message)
            X, Z_samples, P, Omegas_T, Omegas_D = res["X"], res["Z_samples"], res["P"], res["Omegas_T"], res["Omegas_D"]
            if diagnostics is not None:
                loglikelihoods.append(float(res["loglikelihood"]))
                if count > burnin:
                    record_posterior(diagnostics,
                                     posterior_increments(self._expand_Z(Z_samples), self.K, res["X_short"]))
        if diagnostics is not None:
            diagnostics["loglikelihoods"] = loglikelihoods

        return Z_samples, X

    # @profile
    def beam(self, Y: List[np.ndarray], min_u: float = 0., its: int = 100, burnin: int = 50, name: str = 'beamer',
             online: bool = True, sample_U: bool = True,
             force_U: Optional[float] = None, wandb_log: str = "", decay: Optional[float] = None, plot: bool = False,
             plot_folder: str = "./Plots/",
             dump_period: int = 5, dump_path: str = "./", num_of_workers: int = 5,
             actual_Z: Optional[np.ndarray] = None, init_with_pca=True,
             init_iters=0, count=0, init_with_arhmm=True, fast=True,
             arhmm_iters=1000, arhmm_log="all", rotate=False, normalize_emission=False, n_initializations=1,
             init_arhmm_with_kmeans=True,
             freeze_dynamics=False, freeze_transition=False, freeze_duration=False, freeze_emission=False,
             log_file=None, save_space=True, return_accuracy=False,
             skip_duration_message=False, skip_transition_message=False, dyn_V_0_init = None, dyn_Lambda_0_init=None,
             metrics_file=None, diagnostics=None):
        """
        Runs the beam sampling approach for the EDHMM

        Parameters
        ----------
        Y : list
            list of observation sequences

        Optional Parameters
        -------------------
        min_u : scalar (0)
            the minimum auxilliary variable to consider. You can use this to tune
            the algorithm. See the paper for details.
        its : integer (100)
            number of iterations to perform
        burnin : integer (50)
            allow this many iterations before writing samples to disk
        name : string (beamer)
            name of the experiment. This will be prepended to all output files
        online : boolean (True)
            whether or not to run the algo online. Currently this has to be True
        sample_U : boolean (True)
            whether or not to update the auxilliary variable. You probably should leave
            this to True, unless you're poking at the algorithm to see what it does
        updated_D : boolean (True)
            whether or not to update the duration distribution. Again, you should leave
            this to True.
        force_U : None or list of lists (None)
            you can force U to start off from a specific starting place if you like. If
            you set this and set sample_U to False then this auxilliary variable won't
            change throughout the algo.
        min_d : None or list of ints
            You can force a minimum duration per state, if you'd like. You must also set
            max_d if you set min_d.
        max_d : None or list of ints
            maximum duration per state
        """
        self._validate_discrete_sampler_configuration()
        beam_t0 = time.perf_counter()
        pg_warmstart_seconds = 0.0
        if self.nobeam:
            min_u = 0.
            sample_U = False
            force_U = [np.zeros(len(Yi)) for Yi in Y]
            fast = True
        init_with_pca = init_with_pca and self.dynamics.x_dim <= self.emission.obs_dim
        actual_Z_concatenated = None if actual_Z is None else np.concatenate(actual_Z)
        # get support of duration distributions
        # self.set_transition_loglikelihood()

        # sample auxillary variables from some small value
        if force_U:
            U = force_U
        else:
            U = [np.random.uniform(min_u, 0.000000000001, size=len(Yi)) for Yi in Y]

        # get worthy samples given the relaxed U
        alphas = []
        betas = []
        X = []
        if init_with_pca:
            for i, y in enumerate(Y):
                pca = PCA(n_components=self.dynamics.x_dim)
                Xi = pca.fit_transform(y)
                X.append(Xi)
        arhmm_t0 = time.perf_counter()
        if not init_with_arhmm:
            Z_samples_expanded, D_samples_expanded = zip(*[self._random_state_init(len(y)) for y in Y])
            Z_samples = self._merge(Z_samples_expanded, D_samples_expanded)
        else:
            if len(X) == 0:
                D = self.emission.obs_dim
                X_ = Y
                dynamics = LinearGaussianDynamics(K=self.K, x_dim=D)
            else:
                D = self.dynamics.x_dim
                X_ = X
                dynamics = self.dynamics
            duration = DummyDuration(self.K, D)
            initial = LearnableInitial(self.K, duration)
            transition = LoopyTransition(K=self.K, D=D)
            best_dynamics = None
            best_Z_samples = None
            best_loglikelihood = None
            for i in range(n_initializations):
                dynamics_prime = deepcopy(dynamics)
                if dyn_V_0_init is not None:
                    dynamics_prime.V_0 = dyn_V_0_init
                if dyn_Lambda_0_init is not None:
                    dynamics_prime.Lambda_0 = dyn_Lambda_0_init
                arhmm = REDHMM(initial=initial, transition=transition, duration=duration, dynamics=dynamics_prime,
                               nobeam=self.nobeam)
                plot_arhmm = False
                Z_samples, L = arhmm.beam(X_, wandb_log="", init_with_kmeans=init_arhmm_with_kmeans, its=arhmm_iters,
                                          log_prefix="arhmm_init_", plot=plot_arhmm, plot_folder=plot_folder,
                                          dump_period=dump_period, dump_path=dump_path,
                                          )
                loglikelihood = L[-1]
                if (best_loglikelihood is None) or (L[-1] > best_loglikelihood):
                    best_dynamics = dynamics_prime
                    best_Z_samples = Z_samples
                    best_loglikelihood = loglikelihood
            if dyn_V_0_init is not None:
                best_dynamics.params["V_0"] = dynamics.params["V_0"]
            if dyn_Lambda_0_init is not None:
                best_dynamics.params["Lambda_0"] = dynamics.params["Lambda_0"]
            # When PCA is off, the ARHMM above lives in observation space and
            # is used only to seed z/d.  Its dynamics cannot replace the
            # model's latent-space dynamics.
            if len(X) != 0:
                self.dynamics = best_dynamics
            Z_samples = best_Z_samples
            Z_samples_expanded = self._expand_Z(Z_samples)
            if isinstance(self.transition, RecurrentTransition) and len(X) != 0:
                Z_samples_expanded, R_temp = fit_decision_list(Z_samples_expanded, X, self.K, self.transition.affine)
                if self.transition.loopy:
                    self.transition.R = np.repeat(R_temp[np.newaxis], self.transition.R.shape[0], axis=0)
            D_samples_expanded = self._recount_D(Z_samples)
            Z_samples = self._merge(Z_samples_expanded, D_samples_expanded)
            if len(X) != 0:
                self.emission.update(Zs=Z_samples_expanded, Xs=X, Ys=Y, normalize=normalize_emission)
                if rotate:
                    rotation = self.emission.rotate()
                    self.dynamics.rotate(rotation)
                    X = [self.dynamics.rotate_samples(rotation, X_) for X_ in X]

                self._plot_everything(plot_folder, name + "_arhmm_init", count, X[0], np.array(Y[0]), np.array(Y[0]),
                                      Z_samples_expanded[0], self.dynamics.As, D_samples_expanded[0], len(self.states),
                                      wandb_log != "")
        arhmm_init_seconds = time.perf_counter() - arhmm_t0
        Omegas_D = [self.duration.init_pg(len(y)) for y in Y]
        Omegas_T = [self.transition.init_pg(len(y)) for y in Y]

        skip_duration_message = skip_duration_message and (count <= 1500)
        if init_with_pca:
            X_long = X
            X_short = [x[1:] for x in X]
            Z_samples_expanded_short = [z[1:] for z in Z_samples_expanded]
            D_samples_expanded_short = [d[1:] for d in D_samples_expanded]
            self.dynamics.update(Xs=X_long, Zs=[z[1:] for z in Z_samples_expanded])
            self.emission.update(Zs=Z_samples_expanded, Xs=X_long, Ys=Y, normalize=normalize_emission)
            switches = [np.ones_like(d).astype(bool) for d in D_samples_expanded_short]
            if not self.ignore_switches:
                for i, d in enumerate(D_samples_expanded_short):
                    switches[i][1:] = (d == 1)[:-1]
            # switches = [(d == 1)[:-1] for d in D_samples_expanded]
            Psis = [np.random.randn(len(X_) * self.K).reshape((len(X_), self.K)) for X_ in X_short]
            pg_t0 = time.perf_counter()
            for init_it in range(init_iters):
                # for _ in range(2):
                Omegas_T = [self.transition.sample_Omegas(Z_[1:], X_) for Z_, X_ in zip(Z_samples_expanded, X_short)]
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration,
                                                                                DummyDuration) or isinstance(
                    self.duration, NonRecurrentDuration):
                    Omegas_D = [self.duration.sample_Omegas(Z_[1:], X_[:-1], D_[1:]) for Z_, X_, D_ in
                                zip(Z_samples_expanded, X_long, D_samples_expanded)]
                else:
                    raise Exception("Unknown duration")
                self.dynamics.update(Xs=X_long, Zs=Z_samples_expanded_short)
                self.emission.update(Zs=Z_samples_expanded, Xs=X_long, Ys=Y, normalize=normalize_emission)
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                    self.duration.update(Z=Z_samples_expanded_short, X=X_long, D=D_samples_expanded_short,
                                         Omegas=Omegas_D,
                                         switches=switches)
                else:
                    self.duration.update(Z_samples)
                    # raise Exception("Unknown duration")
                if isinstance(self.transition, RecurrentTransition):
                    self.transition.update(Z_samples_expanded_short, X=X_short, Omegas=Omegas_T, switches=switches)
                else:
                    self.transition.update(Z_samples)
                if isinstance(self.transition, RecurrentTransition) and (
                        self.dynamics.x_dim == 2) and self.transition.loopy:
                    X_min = np.min(np.concatenate(X), axis=0)
                    X_max = np.max(np.concatenate(X), axis=0)
                    for k in range(len(self.transition.R)):
                        file_path = os.path.join(plot_folder, f"{name}_init_iter_{init_it}_{k}_decision_boundary.png")
                        self._plot_decision_boundary(X_min, X_max, k,
                                                     title=f"{name} - decision_boundary - {k} - {init_it}",
                                                     fname=file_path)
                    # plt.close(fig)
            pg_warmstart_seconds = time.perf_counter() - pg_t0
            Omegas_D = [np.concatenate([np.mean(omega, axis=0, keepdims=True), omega]) for omega in Omegas_D]
            Omegas_T = [np.concatenate([np.mean(omega, axis=0, keepdims=True), omega]) for omega in Omegas_T]
            self._plot_everything(plot_folder, name + "_init", count, X[0], np.array(Y[0]), np.array(Y[0]),
                                  Z_samples_expanded[0], self.dynamics.As, D_samples_expanded[0], len(self.states),
                                  wandb_log != "")
        X_initialized = len(X) != 0
        if not X_initialized:
            # Seed epsilons for the Kalman draw below. Without this the branch
            # raises UnboundLocalError -- epsilons is referenced here but first
            # assigned after this loop -- so beam(init_with_pca=False) could never
            # run. It cannot simply be hoisted either: the later block builds
            # epsilons from X_long, which is what this draw produces. infer()
            # breaks the same cycle by seeding p/epsilon from a zeros X, so beam()
            # follows infer() rather than inventing a convention.
            epsilons = []
            for i, y in enumerate(Y):
                l = len(Z_samples_expanded[i])
                Xi_seed = np.zeros((y.shape[0], self.dynamics.x_dim))
                epsilons.append(np.zeros(l))
        for i, y in enumerate(Y):
            if not X_initialized:
                Lambdas, Thetas, dyn_thetas = self._backward_kalman_filter(y,
                                                                           Z_samples_expanded[
                                                                               i],
                                                                           D_samples_expanded[
                                                                               i],
                                                                           omegas_dur=Omegas_D[i],
                                                                           omegas_tran=Omegas_T[i],
                                                                           epsilons=epsilons[i],
                                                                           skip_duration_message=skip_duration_message,
                                                                           skip_transition_message=(
                                                                               skip_transition_message
                                                                               or isinstance(
                                                                                   self.transition,
                                                                                   RecurrentTransition,
                                                                               )
                                                                           ))
                Xi = self._forward_kalman_sample(Lambdas, Thetas, Z_samples_expanded[i], dyn_thetas)
                X.append(Xi)
            else:
                if self.extended_kalman:
                    X[i] = np.insert(X[i], 0, X[i][0], axis=0)
                Xi = X[i]
        if self.extended_kalman:
            X_long = X
            X_short = [x[1:] for x in X]
        else:
            X_priors = [self.dynamics.sample_obs(k) for k in self.states]
            X_long = [self._prepend_initial_latent(X_priors[0], x) for x in X]
            X_short = X
        if not X_initialized and isinstance(self.transition, RecurrentTransition):
            Z_samples_expanded, R_temp = fit_decision_list(
                Z_samples_expanded,
                X_short,
                self.K,
                self.transition.affine,
            )
            if self.transition.loopy:
                self.transition.R = np.repeat(
                    R_temp[np.newaxis], self.transition.R.shape[0], axis=0
                )
            Z_samples = self._merge(Z_samples_expanded, D_samples_expanded)
        P = []
        epsilons = []
        for i, y in enumerate(Y):
            Xi = X_long[i]
            l = len(Z_samples_expanded[i])
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                P.append(Xi[:l])
                epsilons.append(np.zeros(l))
            else:
                P.append(Xi[:l])
                epsilons.append(np.zeros(l))
        Psis = [logit(ps) for ps in P]
        # raise Exception("Unknown distribution")
        Omegas_T = [self.transition.sample_Omegas(Z_, X_) for Z_, X_ in zip(Z_samples_expanded, X_short)]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration) or isinstance(
                self.duration, NonRecurrentDuration):
            Omegas_D = [self.duration.sample_Omegas(Z_, X_, D_) for Z_, X_, D_ in
                        zip(Z_samples_expanded, X_long, D_samples_expanded)]
        else:
            raise Exception("Unknown duration")
        # Dotąd git
        # if isinstance(self.duration, DummyDuration):
        switches = [np.ones_like(d).astype(bool) for d in D_samples_expanded]
        if not self.ignore_switches:
            for i, d in enumerate(D_samples_expanded):
                switches[i][1:] = (d == 1)[:-1]
        # else:
        #     switches = [z[:-1] != z[1:] for z in Z_samples_expanded]
        self.emission.update(Zs=Z_samples_expanded, Xs=X_short, Ys=Y, normalize=normalize_emission)
        if rotate:
            rotation = self.emission.rotate()
            self.dynamics.rotate(rotation)
            X = [self.dynamics.rotate_samples(rotation, X_) for X_ in X]
            if self.extended_kalman:
                X_long = X
                X_short = [x[1:] for x in X]
            else:
                # X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                X_long = [self._prepend_initial_latent(X_priors[0], x) for x in X]
                X_short = X
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
            self.duration.update(Z=Z_samples_expanded, X=X_long, D=D_samples_expanded, Omegas=Omegas_D,
                                 switches=switches)
        else:
            self.duration.update(Z_samples)
            # raise Exception("Unknown duration")
        if not freeze_dynamics:
            self.dynamics.update(Xs=X_long, Zs=Z_samples_expanded)
        if isinstance(self.transition, RecurrentTransition):
            self.transition.update(Z_samples_expanded, X=X_short, Omegas=Omegas_T, switches=switches)
        else:
            self.transition.update(Z_samples)
        log.debug('performing inference')
        Z_samples = []
        if online:
            for i, Xi in enumerate(X):
                Yi = Y[i]
                X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                    P_priors = X_priors
                else:
                    P_priors = X_priors
                    # raise Exception("Unknown duration")
                if self.nobeam and self.forward_discrete_states_sample:
                    initial_log_probs = self._initial_augmented_log_probs(
                        X_priors, P_priors, P[i], Xi
                    )
                    betas.append(
                        self.backward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, Y=Yi,
                                         P=P[i], Omegas_D=Omegas_D[i], Omegas_T=Omegas_T[i]))
                    Z_samples.append(
                        self.forward_sample_v2(
                            betas[i][0],
                            betas[i][1],
                            initial_log_probs=initial_log_probs,
                        ))
                elif self.nobeam:
                    alphas.append(
                        self.forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, Y=Yi,
                                        P=P[i], Omegas_D=Omegas_D[i], Omegas_T=Omegas_T[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))
                else:
                    alphas.append(
                        self.beam_forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, Y=Yi, U=U[i], decay=decay,
                                             P=P[i], Omegas_D=Omegas_D[i], Omegas_T=Omegas_T[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))
        else:
            raise NotImplementedError
        # count how many iterations we've done so far
        count = 0
        if init_with_arhmm and len(arhmm_log) > 0:
            count += arhmm_iters
        L = []

        # block gibbs
        Y_est = self.emission.generate_centers(Z_samples_expanded, X)
        self._plot_everything(plot_folder, name, count, X[0], np.array(Y[0]), np.array(Y_est[0]),
                              Z_samples_expanded[0], self.dynamics.As, D_samples_expanded[0], len(self.states),
                              wandb_log != "")
        lower = count + 1
        upper = count + 1 + its
        if metrics_file is not None:
            _append_jsonl(metrics_file, {
                "event": "init",
                "schema_version": 1,
                "nobeam": bool(self.nobeam),
                "forward_discrete_states_sample": bool(self.forward_discrete_states_sample),
                "right": int(self.duration.max_support()[1]),
                "K": int(self.K),
                "n_sequences": len(Y),
                "T_total": int(sum(len(y) for y in Y)),
                "its": int(its),
                "burnin": int(burnin),
                "dump_period": int(dump_period),
                "pid": os.getpid(),
                "arhmm_init_seconds": arhmm_init_seconds,
                "pg_warmstart_seconds": pg_warmstart_seconds,
                "pre_loop_seconds": time.perf_counter() - beam_t0,
            })
        res_accuracies = []
        self._set_latent_scale(X)
        inputs = {"X": X, "Z_samples": Z_samples, "P": P, "Omegas_D": Omegas_D, "Omegas_T": Omegas_T}
        history = []    # start states of the last accepted sweeps, kept off `self` (pool.map pickles the model)
        for count in range(lower, upper):
            # `count` continues from the ARHMM warm-up, so burn-in is measured in main-loop sweeps
            post_burnin = count - lower >= burnin
            plot_ = plot and (((count - 1) % dump_period == 0) or (count == its))
            log_to_file = log_file is not None and (((count - 1) % dump_period == 0) or (count == its))
            log_to_wandb = wandb_log != "" and (((count - 1) % dump_period == 0) or (count == its))
            sweep_t0 = time.perf_counter()
            res = self._redraw_on_divergence(
                lambda st: self._beam(name=name, online=online, count=count, sample_U=sample_U, min_u=min_u,
                                      log_to_wandb=log_to_wandb, Y=Y, decay=decay, plot=plot_,
                                      plot_folder=plot_folder,
                                      burnin=burnin, dump_period=dump_period, dump_path=dump_path,
                                      num_of_workers=num_of_workers, epsilons=epsilons, fast=fast, rotate=rotate,
                                      normalize_emission=normalize_emission,
                                      freeze_dynamics=freeze_dynamics, freeze_transition=freeze_transition,
                                      freeze_duration=freeze_duration, freeze_emission=freeze_emission,
                                      save_space=save_space, skip_duration_message=skip_duration_message,
                                      skip_transition_message=skip_transition_message, **st),
                count, inputs, history)
            sweep_seconds = time.perf_counter() - sweep_t0
            inputs = {k: res[k] for k in self.SWEEP_INPUTS}
            X, Z_samples, P, Omegas_T, Omegas_D = (
                res["X"], res["Z_samples"], res["P"], res["Omegas_T"], res["Omegas_D"]
            )
            Z_expanded = self._expand_Z(Z_samples)
            # one Hungarian alignment per sweep, only when something consumes the score
            if actual_Z is not None and (metrics_file is not None or log_to_wandb or log_to_file):
                Z, scores = self._relabelled_accuracy(Z_expanded, actual_Z_concatenated)
                acc = scores["accuracy"]
            if diagnostics is not None and post_burnin:
                record_posterior(diagnostics,
                                 posterior_increments(Z_expanded, self.K, res["X_short"], res["d_max_t"]))
            if metrics_file is not None:
                sweep_record = {"event": "sweep", "iter": count, "post_burnin": post_burnin,
                                "sweep_seconds": sweep_seconds,
                                "loglikelihood": float(res["loglikelihood"]), "plotted": bool(plot_)}
                if actual_Z is not None:
                    sweep_record.update(scores)
                sweep_record.update(res["sweep_stats"])
                _append_jsonl(metrics_file, sweep_record)
            acc_its = its - 10
            if actual_Z is not None and (log_to_wandb or log_to_file):
                cm = confusion_matrix(Z, actual_Z_concatenated)
                ncm = confusion_matrix(Z, actual_Z_concatenated, normalize='true')
                if log_to_wandb:
                    import wandb
                    log_wandb_scalar_or_array(acc, 'accuracy', step=count)
                    log_wandb_scalar_or_array(cm, 'confusion_matrix', step=count)
                    log_wandb_scalar_or_array(ncm,
                                              'confusion_matrix_normalized', step=count)
                if log_to_file:
                    with open(log_file, "a") as ofile:
                        ofile.write(f"Iter {count} - accuracy: {acc}\n")
                        ofile.write(f"Iter {count} - loglikelihood: {res['loglikelihood']}\n")
                        ofile.write(f"Iter {count} - confusion matrix:\n{cm}\n")
                        ofile.write(f"Iter {count} - normalized cm:\n{ncm}\n\n")
                if not plot:
                    continue
                # Z = self._expand_Z([Z_samples[0]])[0]
                # Z = permute(Z, actual_Z[0], self.K)
                file_path = os.path.join(plot_folder, f"{name}_states.png")
                if count >= acc_its:
                    res_accuracies.append(acc)
                if plot_:
                    plot_states(data_z=actual_Z[0][:1000], z_est=Z[:1000], label="states", fname=file_path)
                if log_to_wandb:
                    wandb.log({"estimated_states": wandb.Image(file_path)}, step=count)
        if return_accuracy:
            return Z_samples, X, np.mean(np.array(res_accuracies))
        return Z_samples, X

    def gen(self, T, init=None):
        """
        generator that yields state/observation tuples

        See Also
        --------
        see EDHMM.sim for more details
        """
        # draw initial state and duration
        if init is None:
            z, d = self.initial.sample()
            z = int(z)
            prev_x = self.dynamics.sample_obs(z)
            d = self.duration.sample_d(z)
        else:
            z, d, prev_x = init
            if isinstance(self.transition, RecurrentTransition):
                z = self.transition.sample_x(i=z, X=prev_x)
            else:
                z = self.transition.sample_x(i=z)
            if d > 1:
                d -= 1
            elif isinstance(self.duration, RecurrentDuration):
                d = self.duration.sample_d(i=z, X=prev_x)
            elif isinstance(self.duration, NonRecurrentDuration):
                d = self.duration.sample_d(state=z)
            else:
                d = self.duration.sample_d(i=z)
        for t in range(T):
            x = self.dynamics.sample_obs(z, prev_x)
            y = self.emission.sample_obs(z, x)
            yield z, x, y, d
            if isinstance(self.transition, RecurrentTransition):
                z = self.transition.sample_x(i=z, X=prev_x)
            else:
                z = self.transition.sample_x(i=z)
            if d > 1:
                d -= 1
            elif isinstance(self.duration, RecurrentDuration):
                d = self.duration.sample_d(i=z, X=prev_x)
            elif isinstance(self.duration, NonRecurrentDuration):
                d = self.duration.sample_d(state=z)
            else:
                d = self.duration.sample_d(i=z)

    # @profile
    def sim(self, T, init=None, runs=1):
        """
        Draws a sequence of length T from the EDHMM

        Parameters
        ----------
        T : int
            number of time points
        """
        Zs, Xs, Ys, Ds = [], [], [], []
        for run in range(runs):
            Z, X, Y, D = [], [], [], []
            for z, x, y, d in self.gen(T, init):
                Z.append(z)
                X.append(x)
                Y.append(y)
                D.append(d)
            Zs.append(np.array(Z, dtype=int))
            Ds.append(np.array(D, dtype=int))
            Xs.append(np.array(X))
            Ys.append(np.array(Y))
        return np.vstack(Zs), np.vstack(Xs), np.vstack(Ys), np.vstack(Ds)
