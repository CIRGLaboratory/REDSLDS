import os
import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from phdstuff.EDSLDS_numpy.utils import worker_pool as Pool
from scipy.special import logsumexp, logit
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

from phdstuff.EDSLDS_numpy.duration import RecurrentDuration, DummyDuration, NonRecurrentDuration
from phdstuff.EDSLDS_numpy.dynamics import LinearGaussianDynamics
from phdstuff.EDSLDS_numpy.edhmm import log, Categorical
from phdstuff.EDSLDS_numpy.initial import AbstractInitial
from phdstuff.EDSLDS_numpy.plot import plot_actual_shifts, plot_observations, \
    plot_phase_portrait, plot_durations
from phdstuff.EDSLDS_numpy.transition import RecurrentTransition, LoopyTransition
from phdstuff.plotting import plot_states
from phdstuff.utils import permute, log_wandb_scalar_or_array
from sklearn.inspection import DecisionBoundaryDisplay

def _logsumexp(a, b):
    return logsumexp(np.stack([a, b]), axis=0)


class REDHMM(object):
    def __init__(self, initial: AbstractInitial, transition: RecurrentTransition,
                 dynamics: LinearGaussianDynamics, duration, extended_kalman=False, nobeam=False, ignore_switches=False, forward_discrete_states_sample=False):
        self.initial = initial
        self.transition = transition
        self.dynamics = dynamics
        self.duration = duration
        self.K = len(initial)
        self.states = range(self.K)
        self.extended_kalman = extended_kalman
        self.nobeam = nobeam
        self.affine = transition.affine or duration.affine or dynamics.affine
        self.ignore_switches = ignore_switches
        self.forward_discrete_states_sample = forward_discrete_states_sample

    def dump(self, name, location="./"):
        self.initial.dump(location, name)
        self.transition.dump(location, name)
        self.dynamics.dump(location, name)
        self.duration.dump(location, name)

    # @profile

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

    def loglikelihood(self, Zs, Xs, Ps, Omegas_T=None, Omegas_D=None):  # TODO przeorać tę funkcję
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
        for Z, X, P, Omegas_t, Omegas_d in zip(Zs, Xs, Ps, Omegas_T, Omegas_D):
            # assert (len(X) > len(Omegas_t))
            l += self.initial.loglikelihood(z=Z[0], p=P[0], x=X[0])
            for t in range(1, len(Z)):
                i = Z[t - 1][0]
                j = Z[t][0]
                di = Z[t - 1][1]
                x = X[t+1]
                p = P[t]
                prev_x = X[t]
                omega_d = omega_t = None
                if Omegas_t is not None:
                    omega_t = Omegas_t[t - 1]
                if Omegas_d is not None:
                    omega_d = Omegas_d[t]
                if (i == j) and not loopy:
                    l += self.dynamics.loglikelihood(j, prev_x, x)
                else:
                    if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                        l += self.duration.loglikelihood(i, di, prev_x, omega=omega_d)
                    else:
                        l += self.duration.loglikelihood(i, di)
                    if isinstance(self.transition, RecurrentTransition):
                        l += self.transition.loglikelihood(prev_x, i, j, omega=omega_t)
                    else:
                        l += self.transition.loglikelihood(i, j)
                    l += (
                        self.dynamics.loglikelihood(j, prev_x, x))
        return l

    def backward_v2(self, X_priors, P_priors, P, X, Omegas_T=None, Omegas_D=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """

        log.info('running forward algorithm')
        # initialise alphahat
        limit = len(X) - 1 if self.extended_kalman else len(X)
        left, right = self.duration.max_support()
        betahat = np.log(np.zeros((limit, right, self.K)))
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
                                                           previous_observation=prev_x)
                prev_x = x
        # ol = np.maximum(-1000000000000 * np.ones_like(ol), ol)
        log.debug('starting iteration')
        for t, y in reversed(enumerate(Y)):
            t_ = t if not self.extended_kalman else t + 1
            if t == limit - 1:
                betahat[limit - 1] = np.zeros_like(betahat[limit - 1])
            else:
                tran_ll = np.zeros((self.K, self.K))
                dur_ll = np.zeros((self.K, right))
                for i in self.states:
                    if isinstance(self.transition, RecurrentTransition):
                        assert len(X) > len(Omegas_T)
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
                    betahat[t, 0, :] = _logsumexp(logsumexp(betahat[t + 1, :, i] + log_prob.T,
                                                            axis=1), betahat[t, 0, :].reshape((1, self.K)))
                    switch_lls[t, :, i] = log_prob
                switch_lls[t-1] += ol[t_].T[np.newaxis]
            betahat[t] += ol[t_]
                # betahat[t] -= logsumexp(betahat[t])
        return betahat, switch_lls


    def forward_v2(self, X_priors, P_priors, P, X, Omegas_T=None, Omegas_D=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """
        log.info('running forward algorithm')
        # initialise alphahat
        limit = len(X) - 1 if self.extended_kalman else len(X)
        left, right = self.duration.max_support()
        alphahat = np.log(np.zeros((limit, right, self.K)))
        switch_lls = np.zeros((limit, self.K, self.K, right))
        log.debug('calculating observation loglikelihoods')
        ol = np.zeros((len(X), right, self.K))
        for i in self.states:
            if not self.extended_kalman:
                prev_x = X_priors[i]
            else:
                prev_x = X[0]
            for t, y in enumerate(X):
                t_ = t if not self.extended_kalman else t + 1
                x = X[t_]
                ol[t, :, i] = self.dynamics.loglikelihood(state=i, observation=x,
                                                          previous_observation=prev_x)
                prev_x = x
        # ol = np.maximum(-1000000000000 * np.ones_like(ol), ol)
        log.debug('starting iteration')

        for t, y in enumerate(X):
            t_ = t if not self.extended_kalman else t + 1
            x = X[t_]
            if t == 0:
                for i in self.states:
                    P0 = np.minimum(np.maximum(P_priors[i] if not self.extended_kalman else P[0][i], 0.00001),
                                    0.99999)  # if not self.extended_kalman else P[0][i]
                    X0 = X_priors[i]  # if not self.extended_kalman else X[0]
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
                        assert len(X) > len(Omegas_T)
                        tran_ll[i] = self.transition.get_log_transition(i, X[t_ - 1], omega=Omegas_T[t - 1])
                    else:
                        tran_ll[i] = self.transition.loglikelihood(i, self.states)
                    if isinstance(self.duration, RecurrentDuration):
                        dur_ll[i] = self.duration.get_log_transition(i, X[t_ - 1], omega=Omegas_D[t_ - 1])
                    else:
                        dur_ll[i] = self.duration.loglikelihood(i, np.arange(right) + 1)
                # ic(dur_ll)
                assert not np.any(np.isnan(alphahat)), alphahat
                assert not np.any(np.isnan(dur_ll)), dur_ll
                assert not np.any(np.isnan(tran_ll)), tran_ll
                log_probs = tran_ll[:,:,np.newaxis] + dur_ll[np.newaxis, :, :]  # TODO einsum
                assert not np.any(np.isnan(log_probs)), log_probs
                alphahat[t, :-1, :] = _logsumexp(alphahat[t, :-1, :], alphahat[t - 1, 1:, :])
                for i in self.states:
                    log_prob = log_probs[:, i]
                    alphahat[t, :, i] = _logsumexp(alphahat[t, :, i],
                                                   logsumexp(alphahat[t - 1, 0, :].reshape((1, self.K)) + log_prob.T,
                                                             axis=1))
                    switch_lls[t - 1, :, i] = log_prob
                switch_lls[t-1] += ol[t_].T[np.newaxis]
            assert not np.any(np.isnan(ol[t_])), ol[t_]
            alphahat[t] = alphahat[t] + ol[t_]
            assert not np.any(np.isnan(ol[t_])), ol[t_]
            assert not np.any(np.isnan(alphahat[t])), alphahat[t]

        return alphahat, switch_lls

    def beam_forward_v2(self, X_priors, P_priors, P, X, U, W=None, decay=None, Omegas_T=None, Omegas_D=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """
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
        for i in self.states:
            if not self.extended_kalman:
                prev_x = X_priors[i]
            else:
                prev_x = X[0]
            for t, y in enumerate(X):
                t_ = t if not self.extended_kalman else t + 1
                x = X[t_]
                ol[t_, :, i] = self.dynamics.loglikelihood(state=i, observation=x,
                                                           previous_observation=prev_x)

                prev_x = x
        # ol = np.maximum(-1000000000000 * np.ones_like(ol), ol)
        log.debug('starting iteration')

        for t, y in enumerate(X):
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
                tran_ll = np.zeros((self.K, self.K))
                dur_ll = np.zeros((self.K, right))
                for i in self.states:
                    if isinstance(self.transition, RecurrentTransition):
                        tran_ll[i] = self.transition.get_log_transition(i, X[t_ - 1], omega=Omegas_T[t - 1])
                    else:
                        tran_ll[i] = self.transition.loglikelihood(i, self.states)
                    if isinstance(self.duration, RecurrentDuration):
                        dur_ll[i] = self.duration.get_log_transition(i, X[t_ - 1], omega=Omegas_D[t_ - 1])
                    else:
                        dur_ll[i] = self.duration.loglikelihood(i, np.arange(right) + 1)

                u = U[t]
                log_probs = tran_ll[:,:,np.newaxis] + dur_ll[np.newaxis, :, :]  # TODO einsum
                alphahat[t, :-1, :] = _logsumexp(alphahat[t, :-1, :], alphahat[t - 1, 1:, :])
                for i in self.states:
                    log_prob = log_probs[:, i]
                    prob = np.exp(log_prob)  # TODO einsum
                    reachable = prob.T > u
                    log_mask = np.log(reachable.astype(float))
                    switch_lls[t - 1, :, i] = log_mask.T
                    alphahat[t, :, i] = _logsumexp(alphahat[t, :, i],
                                                   logsumexp(alphahat[t - 1, 0, :] + log_mask, axis=1))
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

    def beam_backward_sample_v2(self, alphahat, reachable):
        T = len(alphahat)
        try:
            Z = [self._sample_z(alphahat[-1].T, reachable[-1].T)]
        except ValueError:
            print(alphahat[-1])
            raise
        ones_mask = np.zeros_like(reachable[0])
        ones_mask[0, :] = True
        for t in reversed(range(T - 1)):
            # t_ = t if not self.extended_kalman else t + 1
            mask = ones_mask.copy()
            s, d = Z[-1]
            if d < mask.shape[0]:
                mask[d, s] = True
            a = alphahat[t]
            reach = reachable[t] & mask
            z = self._sample_z(a.T, reach.T)
            Z.append(z)
        Z.reverse()
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
            a[0, :] += switch_lls[t, :, s, d - 1]
            z = self._sample_z(a.T, mask.T)
            Z.append(z)
        Z.reverse()
        return Z

    def forward_sample_v2(self, betahat, switch_lls):
        T = len(betahat)
        try:
            Z = [self._sample_z(betahat[0].T, np.ones_like(betahat[0].T).astype(bool))]
        except ValueError:
            print(betahat[0])
            raise
        for t in range(T - 1):
            s, d = Z[-1]
            if d > 1:
                Z.append((s, d - 1))
                continue
            a = betahat[t]
            a = a + switch_lls[t - 1]
            z = self._sample_z(a.T, np.ones_like(a).T.astype(bool))
            Z.append(z)
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

    def _beam(self, name, online, count, sample_U, Z_samples, min_u, log_to_wandb, P, epsilons, X, Omegas_D,
              Omegas_T, decay,
              update_D, plot, plot_folder, prev_X, burnin, dump_period, dump_path, num_of_workers, actual_Z,
              cache, double_sample=False, init_with_kmeans=False, init_iters=0, fast=True, save_space=True,
              skip_duration_message=False, skip_transition_message=False):
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
        log.debug('slice sample took %ss' % (time.time() - start))

        # states
        start = time.time()
        if online:
            with Pool(num_of_workers, len(X)) as pool:
                if self.forward_discrete_states_sample:
                    betas = pool.map(
                        self._betas_generator(P, X, U, X_priors=X_priors, P_priors=P_priors, Omegas_T=Omegas_T,
                                               Omegas_D=Omegas_D, decay=decay, fast=fast),
                        range(len(X)))
                    Z_samples = pool.map(
                        self._Zs_generator(betas),
                        range(len(X)))
                else:
                    alphas = pool.map(
                        self._alphas_generator(P, X, U, X_priors=X_priors, P_priors=P_priors, Omegas_T=Omegas_T,
                                               Omegas_D=Omegas_D, decay=decay, fast=fast),
                        range(len(X)))
                    Z_samples = pool.map(
                        self._Zs_generator(alphas),
                        range(len(X)))
            log.debug('inference took %ss' % (time.time() - start))
        else:
            raise NotImplementedError
        Z_samples_expanded = self._expand_Z(Z_samples)
        D_samples_expanded = self._expand_D(Z_samples)
        Z_for_update = [z[1:] for z in Z_samples_expanded]
        D_for_update = [d[1:] for d in D_samples_expanded]
        if plot:
            self._plot_everything(plot_folder, name, count, X[0],
                                  Z_samples_expanded[0], self.dynamics.As, D_samples_expanded[0], len(self.states),
                                  log_to_wandb, save_space=save_space)

        X_long = X
        X_short = [x[1:] for x in X]
        Omegas_T = [self.transition.sample_Omegas(Z_, X_) for Z_, X_ in zip(Z_for_update, X_short)]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration) or isinstance(
                self.duration, NonRecurrentDuration):
            Omegas_D = [self.duration.sample_Omegas(Z_, X_, D_) for Z_, X_, D_ in
                        zip(Z_for_update, X_long, D_for_update)]
        else:
            raise Exception("Unknown duration")
        # if wandb_log != "":
        #     log_wandb_scalar_or_array(np.concatenate(Z_samples_expanded), "Z_samples_expanded", step=count)
        assert not np.any(np.isnan(np.concatenate(Omegas_T))), f"Omegas_T: {Omegas_T}"
        assert not np.any(np.isnan(np.concatenate(Omegas_D))), f"Omegas_D: {Omegas_D}"

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
        switches_for_update = [s[1:] for s in switches]
        # switches = [(d == 1)[:-1] for d in D_samples_expanded]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
            self.duration.update(Z=Z_for_update, X=X_long, D=D_for_update, Omegas=Omegas_D,
                                 switches=switches_for_update)
        elif isinstance(self.duration, NonRecurrentDuration):
            self.duration.update(Z_samples)
        else:
            raise Exception("Unknown duration")
        if isinstance(self.transition, RecurrentTransition):
            self.transition.update(Z_for_update, X_short, Omegas_T, switches=switches_for_update)
        else:
            self.transition.update(Z_samples)
        # loglikelihood
        self.dynamics.update(Xs=X_long, Zs=Z_for_update)
        l = self.loglikelihood(Zs=[z[1:] for z in Z_samples], Xs=X, Ps=P)
        if log_to_wandb:
            import wandb
            self.duration.log_to_wandb(count)
            self.transition.log_to_wandb(count)
            self.dynamics.log_to_wandb(count)
            wandb.log({"loglikelihood": l}, step=count)
        # L.append(l)
        log.info("log loglikelihood at iteration %s: %s" % (count, l))

        if count > burnin:
            if count % dump_period == 0:
                log.debug('writing iteration %s to disk' % count)
                self.dump(name, dump_path)
        return {"prev_X": prev_X,
                "X": X,
                "Z_samples": Z_samples,
                "P": P,
                "Omegas_T": Omegas_T,
                "Omegas_D": Omegas_D,
                "loglikelihood": l
                }

    def _alphas_generator(self, P, X, U, X_priors, P_priors, decay, fast, Omegas_T=None, Omegas_D=None):
        if self.nobeam:
            U = [np.zeros_like(u) for u in U]

        def _alphas_pass(i):
            Xi = X[i]
            Pi = P[i]
            Omegas_Ti = Omegas_T[i]
            Omegas_Di = Omegas_D[i]
            if self.nobeam:
                return self.forward_v2(X_priors=X_priors, P_priors=P_priors, P=Pi, X=Xi, Omegas_T=Omegas_Ti,
                                       Omegas_D=Omegas_Di)
            else:
                return self.beam_forward_v2(X_priors=X_priors, P_priors=P_priors, P=Pi, X=Xi, U=U[i], decay=decay,
                                            Omegas_T=Omegas_Ti, Omegas_D=Omegas_Di)

        return _alphas_pass

    def _betas_generator(self, P, X, U, X_priors, P_priors, decay, fast, Omegas_T=None, Omegas_D=None):
        if self.nobeam:
            U = [np.zeros_like(u) for u in U]

        def _betas_pass(i):
            Xi = X[i]
            Pi = P[i]
            Omegas_Ti = Omegas_T[i]
            Omegas_Di = Omegas_D[i]
            if self.nobeam:
                return self.backward_v2(X_priors=X_priors, P_priors=P_priors, P=Pi, X=Xi, Omegas_T=Omegas_Ti,
                                       Omegas_D=Omegas_Di)
            else:
                raise NotImplementedError

        return _betas_pass

    def _Zs_generator(self, alphas):
        def _alphas_pass(i):
            return self.backward_sample_v2(alphas[i][0], alphas[i][1])

        def _betas_pass(i):
            return self.backward_sample_v2(alphas[i][0], alphas[i][1])

        if self.forward_discrete_states_sample:
            return _betas_pass
        return _alphas_pass

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
    def _plot_everything(self, path, name, iter, X, Z, As, Dseq, states_n, log_to_wandb, save_space=True):
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
        file_path = os.path.join(path, f"{name}_durations.png")
        fig, ax = plot_durations(Z, Dseq, states_n, title=f"{name} - durations - {iter}")
        fig.savefig(file_path)
        plt.close(fig)
        if log_to_wandb:
            plot_dict["durations"] = wandb.Image(file_path)
        if log_to_wandb:
            wandb.log(plot_dict, step=iter)
        plt.close('all')

    def _count(self, Zs):
        res = []
        for Z in Zs:
            tmp = np.zeros_like(Z)
            prev_z = None
            for i in range(len(Z) - 1, -1, -1):
                z = Z[i]
                tmp[i] = 1 if prev_z != z else tmp[i + 1] + 1
                prev_z = z
            res.append(tmp)
        return res

    def _merge(self, Zs, Ds):
        res = []
        for Z, D in zip(Zs, Ds):
            res.append(list(zip(Z, D)))
        return res

    def infer(self,X, min_u: float = 0., its: int = 100,
              online: bool = True, sample_U: bool = True,
              force_U: Optional[float] = None, decay: Optional[float] = None,
              num_of_workers: int = 5,
              init_iters=0, count=0, fast=True, skip_duration_message=False, skip_transition_message=False):
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
        if self.nobeam:
            min_u = 0.
            sample_U = False
            force_U = [np.zeros(len(Xi)) for Xi in X]
            fast = True
        # get support of duration distributions
        # self.set_transition_loglikelihood()

        # sample auxillary variables from some small value
        if force_U:
            U = force_U
        else:
            U = [np.random.uniform(min_u, 0.000000000001, size=len(Xi)) for Xi in X]

        # get worthy samples given the relaxed U
        alphas = []
        Z_samples_expanded, D_samples_expanded = zip(*[self._random_state_init(len(x)) for x in X])
        Omegas_D = [self.duration.init_pg(len(x)) for x in X]
        Omegas_T = [self.transition.init_pg(len(x)) for x in X]
        P = []
        epsilons = []
        for i, Xi in enumerate(X):
            l = len(Z_samples_expanded[i])
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                P.append(Xi[:l])
                epsilons.append(np.zeros(l))
            else:
                P.append(Xi[:l])
                epsilons.append(np.zeros(l))
        P = []
        epsilons = []
        for i, Xi in enumerate(X):
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
                X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                    P_priors = X_priors
                else:
                    P_priors = X_priors
                    # raise Exception("Unknown duration")
                if self.nobeam and self.forward_discrete_states_sample:
                    betas.append(
                        self.backward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi,
                                        P=P[i], Omegas_T=Omegas_T[i], Omegas_D=Omegas_D[i]))
                    Z_samples.append(
                        self.forward_sample_v2(alphas[i][0], alphas[i][1]))
                elif self.nobeam:
                    alphas.append(
                        self.forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi,
                                        P=P[i], Omegas_T=Omegas_T[i], Omegas_D=Omegas_D[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))
                else:
                    alphas.append(
                        self.beam_forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, U=U[i], decay=decay,
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
        for count in range(lower, upper):
            res = self._infer(online=online, count=count, sample_U=sample_U, Z_samples=Z_samples, min_u=min_u,
                              X=X, decay=decay,
                              num_of_workers=num_of_workers, Omegas_D=Omegas_D, Omegas_T=Omegas_T, P=P,
                              epsilons=epsilons, fast=fast, skip_duration_message=skip_duration_message,
                              skip_transition_message=skip_transition_message)
            X, Z_samples, P, Omegas_T, Omegas_D = res["X"], res["Z_samples"], res["P"], res["Omegas_T"], res["Omegas_D"]

        return Z_samples, X

    # @profile
    def beam(self, X, min_u: float = 0., its: int = 100, burnin: int = 50, name: str = 'beamer',
             online: bool = True, sample_U: bool = True, update_D: bool = True,
             force_U: Optional[float] = None, wandb_log: str = "", decay: Optional[float] = None, plot: bool = False,
             plot_folder: str = "./Plots/",
             dump_period: int = 5, dump_path: str = "./", num_of_workers: int = 5,
             actual_Z: Optional[np.ndarray] = None, double_sample: bool = False, cache=None, init_with_kmeans=False,
             init_iters=100, fast=True, log_prefix="", log_file=None, save_space=True):
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
        if self.nobeam:
            min_u = 0.
            sample_U = False
            force_U = [np.zeros(len(Xi)) for Xi in X]
            fast = True
        actual_Z_concatenated = None if actual_Z is None else np.concatenate(actual_Z)
        # get support of duration distributions
        # self.set_transition_loglikelihood()

        # sample auxillary variables from some small value
        if force_U:
            U = force_U
        else:
            U = [np.random.uniform(min_u, 0.000000000001, size=len(Xi)) for Xi in X]

        # get worthy samples given the relaxed U
        alphas = []
        X_short = [x[1:] for x in X]
        X_long = X
        Omegas_D = [self.duration.init_pg(len(x)) for x in X_short]
        Omegas_T = [self.transition.init_pg(len(x)) for x in X_short]
        if init_with_kmeans:
            kmeans = KMeans(n_clusters=self.K).fit(np.concatenate(X))
            Z_samples = [kmeans.predict(x) for x in X]
            if not isinstance(self.duration, DummyDuration):
                D_samples = self._count(Z_samples)
            else:
                D_samples = [np.ones_like(Z) for Z in Z_samples]
            Z = self._merge(Z_samples, D_samples)
            Z_for_update = [z[1:] for z in Z_samples]
            D_for_update = [d[1:] for d in D_samples]
            switches = [np.ones_like(d).astype(bool) for d in D_samples]
            switches_for_update = [s[1:] for s in switches]
            if not self.ignore_switches:
                for i, d in enumerate(D_samples):
                    switches[i][1:] = (d == 1)[:-1]
            Psis = [np.random.randn(len(X_) * self.K).reshape((len(X_), self.K)) for X_ in X_short]
            for init_it in range(init_iters):
                Omegas_T = [self.transition.sample_Omegas(Z_[1:], X_) for Z_, X_ in zip(Z_samples, X_short)]
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration,
                                                                                DummyDuration) or isinstance(
                    self.duration, NonRecurrentDuration):
                    Omegas_D = [self.duration.sample_Omegas(Z_[1:], X_[:-1], D_[1:]) for Z_, X_, D_ in
                                zip(Z_samples, X_long, D_samples)]
                else:
                    raise Exception("Unknown duration")
                self.dynamics.update(Xs=X_long, Zs=Z_for_update)
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
                    self.duration.update(Z=Z_for_update, X=X_long, D=D_samples, Omegas=Omegas_D,
                                         switches=switches)
                elif isinstance(self.duration, NonRecurrentDuration):
                    self.duration.update(Z=Z)
                else:
                    raise Exception("Unknown duration")
                if isinstance(self.transition, RecurrentTransition):
                    self.transition.update(Z_for_update, X=X_short, Omegas=Omegas_T, switches=switches_for_update)
                else:
                    self.transition.update(Z)
        else:
            Z_samples, D_samples = zip(*[self._random_state_init(len(x)) for x in X])
            Z = self._merge(Z_samples, D_samples)
            Z_for_update = [z[1:] for z in Z_samples]
            D_for_update = [d[1:] for d in D_samples]
        P = []
        epsilons = []
        for i, Xi in enumerate(X):
            l = len(Z_samples[i])
            if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, NonRecurrentDuration):
                P.append(Xi[-l:])
                epsilons.append(np.zeros(l))
            else:
                P.append(Xi[-l:])
                epsilons.append(np.zeros(l))
        Psis = [logit(ps) for ps in P]
        # raise Exception("Unknown distribution")
        Omegas_T = [self.transition.sample_Omegas(Z_, X_) for Z_, X_ in zip(Z_for_update, X_short)]
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration) or isinstance(
                self.duration, NonRecurrentDuration):
            Omegas_D = [self.duration.sample_Omegas(Z_, X_, D_) for Z_, X_, D_ in
                        zip(Z_for_update, X_long, D_for_update)]
        else:
            raise Exception("Unknown duration")
        # Dotąd git
        # if isinstance(self.duration, DummyDuration):
        switches = [np.ones_like(d).astype(bool) for d in D_samples]
        if not self.ignore_switches:
            for i, d in enumerate(D_samples):
                switches[i][1:] = (d == 1)[:-1]
        # else:
        #     switches = [z[:-1] != z[1:] for z in Z_samples]
        Z_for_update = [z[1:] for z in Z_samples]
        D_for_update = [d[1:] for d in D_samples]
        switches_for_update = [s[1:] for s in switches]
        self.dynamics.update(Xs=X_long, Zs=Z_for_update)
        if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, DummyDuration):
            self.duration.update(Z=Z_for_update, X=X_long, D=D_for_update, Omegas=Omegas_D,
                                 switches=switches_for_update)
        elif isinstance(self.duration, NonRecurrentDuration):
            self.duration.update(Z=Z)
        else:
            raise Exception("Unknown duration")
        if isinstance(self.transition, RecurrentTransition):
            self.transition.update(Z_for_update, X=X_short, Omegas=Omegas_T, switches=switches_for_update)
        else:
            self.transition.update(Z)
        log.debug('performing inference')
        Z_samples = []
        if online:
            for i, Xi in enumerate(X):
                X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                if isinstance(self.duration, RecurrentDuration) or isinstance(self.duration, NonRecurrentDuration):
                    P_priors = X_priors
                else:
                    P_priors = X_priors
                    # raise Exception("Unknown duration")
                if self.nobeam and self.forward_discrete_states_sample:
                    betas.append(
                        self.backward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi,
                                        P=P[i], Omegas_D=Omegas_D[i], Omegas_T=Omegas_T[i]))
                    Z_samples.append(
                        self.forward_sample_v2(alphas[i][0], alphas[i][1]))

                elif self.nobeam:
                    alphas.append(
                        self.forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi,
                                        P=P[i], Omegas_D=Omegas_D[i], Omegas_T=Omegas_T[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))

                else:
                    alphas.append(
                        self.beam_forward_v2(X_priors=X_priors, P_priors=P_priors, X=Xi, U=U[i], decay=decay,
                                             P=P[i], Omegas_D=Omegas_D[i], Omegas_T=Omegas_T[i]))
                    Z_samples.append(
                        self.backward_sample_v2(alphas[i][0], alphas[i][1]))
        else:
            raise NotImplementedError
        # count how many iterations we've done so far
        count = 0

        L = []

        # block gibbs
        prev_X = None
        for count in range(its):
            log_to_file = log_file is not None and (((count - 1) % dump_period == 0) or (count == its))

            log_to_wandb = wandb_log != "" and ((count % dump_period == 0) or (count == its - 1))
            res = self._beam(name=log_prefix + name, online=online, count=count, sample_U=sample_U, Z_samples=Z_samples,
                             min_u=min_u, epsilons=epsilons,
                             log_to_wandb=log_to_wandb, X=X, decay=decay, update_D=update_D, plot=plot,
                             plot_folder=plot_folder, prev_X=prev_X,
                             burnin=burnin, dump_period=dump_period, dump_path=dump_path,
                             num_of_workers=num_of_workers, actual_Z=actual_Z, cache=cache,
                             double_sample=double_sample, Omegas_D=Omegas_D, Omegas_T=Omegas_T, P=P,
                             init_iters=init_iters, fast=fast, save_space=save_space)
            L.append(res["loglikelihood"])
            prev_X, X, Z_samples, P, Omegas_T, Omegas_D = res["prev_X"], res["X"], res["Z_samples"], res["P"], res[
                "Omegas_T"], res["Omegas_D"]
            if actual_Z is not None:
                Z = np.concatenate(self._expand_Z(Z_samples))
                try:
                    Z = permute(Z, actual_Z_concatenated, self.K)
                except:
                    print("permutation failed")
                acc = accuracy_score(Z, actual_Z_concatenated)
                cm = confusion_matrix(Z, actual_Z_concatenated)
                ncm = confusion_matrix(Z, actual_Z_concatenated, normalize='true')
                if log_to_wandb:
                    import wandb
                    log_wandb_scalar_or_array(accuracy_score(Z, actual_Z_concatenated), 'accuracy', step=count)
                    log_wandb_scalar_or_array(confusion_matrix(Z, actual_Z_concatenated), 'confusion_matrix',
                                              step=count)
                    log_wandb_scalar_or_array(confusion_matrix(Z, actual_Z_concatenated, normalize='true'),
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
                plot_states(data_z=actual_Z[0][:1000], z_est=Z[:1000], label="states", fname=file_path)
                if log_to_wandb:
                    wandb.log({"estimated_states": wandb.Image(file_path)}, step=count)
        return Z_samples, L

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
                z = self.transition.sample_x(i=z, X=prev_x)
                d = self.duration.sample_d(i=z, X=prev_x)
            elif isinstance(self.duration, NonRecurrentDuration):
                d = self.duration.sample_d(state=z)
            else:
                z = self.transition.sample_x(i=z)
                d = self.duration.sample_d(i=z)
        for t in range(T):
            x = self.dynamics.sample_obs(z, prev_x)
            yield z, x, d
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
        Zs, Xs, Ds = [], [], []
        for run in range(runs):
            Z, X, D = [], [], []
            for z, x, d in self.gen(T, init):
                Z.append(z)
                X.append(x)
                D.append(d)
            Zs.append(np.array(Z, dtype=int))
            Ds.append(np.array(D, dtype=int))
            Xs.append(np.array(X))
        return np.vstack(Zs), np.vstack(Xs), np.vstack(Ds)
