import os
import time

import numpy as np
import wandb
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

from .duration import TestPoisson
from .dynamics import AbstractDynamics, LinearGaussianDynamics
from .edhmm import log, Categorical
from .emission import AbstractEmission, LinearGaussianEmission
from .initial import AbstractInitial, Initial, LearnableInitial
from .log_space import elnsum
from .plot import plot_actual_shifts, plot_observations, plot_phase_portrait, plot_emission, \
    plot_durations, plot_distribution_shift
from .transition import Transition
from .utils import invert, turn_matrix_positive_semidefinite, \
    aggresively_turn_matrix_positive_semidefinite
from phdstuff.plotting import plot_states
from phdstuff.utils import permute, log_wandb_scalar_or_array
from .utils import worker_pool as Pool


class EDSLDS:
    def __init__(self, initial: AbstractInitial, transition: Transition, emission: AbstractEmission,
                 dynamics: AbstractDynamics, duration: TestPoisson):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.dynamics = dynamics
        self.duration = duration
        self.K = len(initial)
        self.states = range(self.K)

    def dump(self, name, location="./"):
        self.initial.dump(location, name)
        self.transition.dump(location, name)
        self.emission.dump(location, name)
        self.dynamics.dump(location, name)
        self.duration.dump(location, name)

    @staticmethod
    def load(path, name, transition_type=Transition, emission_type=LinearGaussianEmission,
             dynamics_type=LinearGaussianDynamics, initial_type=Initial, duration_type=TestPoisson):
        transition = transition_type.load(path, name)
        emission = emission_type.load(path, name)
        dynamics = dynamics_type.load(path, name)
        duration = duration_type.load(path, name)
        if initial_type == LearnableInitial:
            initial = initial_type.load(path, name, duration)
        else:
            initial = initial_type.load(path, name)
        return EDSLDS(initial, transition, emission, dynamics, duration)

    # @profile
    def _backward_kalman_filter(self, obs, states):
        assert isinstance(states, np.ndarray)
        states = states.astype(int)
        Cs = self.emission.get_transitions(states)
        if self.dynamics.affine:
            Cs = np.concatenate([Cs, np.zeros((Cs.shape[0], Cs.shape[1], 1))], axis=2)
        eSigmas = self.emission.get_sigmas(states)
        Ds = self.dynamics.get_transitions(states)
        dSigmas = self.dynamics.get_sigmas(states)
        T = states.shape[0]
        Lambdas = np.zeros((T, dSigmas.shape[1], dSigmas.shape[2]))
        Thetas = np.zeros((T, dSigmas.shape[1]))
        # Batch invert all emission and dynamics covariances at once
        R_invs = np.linalg.inv(eSigmas)
        dSigma_invs = np.linalg.inv(dSigmas)
        Lambdas[T - 1] = Cs[T - 1].T @ R_invs[T - 1] @ Cs[T - 1]
        # if np.all(obs[T-1] != np.nan):
        Thetas[T - 1] = Cs[T - 1].T @ R_invs[T - 1] @ obs[T - 1]
        for t in range(T - 2, -1, -1):
            # Compute - use pre-computed dSigma inverse
            dSigma_inv_t1 = dSigma_invs[t + 1]
            J = Lambdas[t + 1] @ invert(Lambdas[t + 1] + dSigma_inv_t1)
            L = np.eye(J.shape[0], J.shape[1]) - J
            # Predict - reuse dSigma_inv_t1
            Lambda_1 = Ds[t + 1].T @ (L @ Lambdas[t + 1] @ L.T + J @ dSigma_inv_t1 @ J.T) @ Ds[t + 1]
            Theta_1 = Ds[t + 1].T @ L @ (Thetas[t + 1])
            # Update
            if t > 0:
                Lambdas[t] = Lambda_1 + Cs[t].T @ R_invs[t] @ Cs[t]
                # Thetas[t] = Theta_1
                # if np.all(obs[t] != np.nan):
                #     Thetas[t] += Cs[t].T @ R_invs[t] @ obs[t]
                Thetas[t] = Theta_1 + Cs[t].T @ R_invs[t] @ obs[t]
            else:
                Lambdas[t] = Lambda_1
                Thetas[t] = Theta_1
        return Lambdas, Thetas

    # @profile
    def _forward_kalman_sample(self, Lambdas, Thetas, states):
        assert isinstance(states, np.ndarray)
        assert isinstance(Lambdas, np.ndarray)
        assert isinstance(Thetas, np.ndarray)

        states = states.astype(int)
        T = Lambdas.shape[0]
        As = self.dynamics.get_transitions()
        Sigmas = self.dynamics.get_sigmas()
        # Pre-compute all Sigma inverses (one per state)
        Sigma_invs = np.linalg.inv(Sigmas)
        res = np.zeros((T, As.shape[1]))
        x = self.dynamics.sample_obs(0, None) # TODO proper init
        if self.dynamics.affine:
            x = np.concatenate([x, np.ones(1)])
        for t in range(T):
            state = states[t]
            Sigma_inv = Sigma_invs[state]
            S = invert(Sigma_inv + Lambdas[t])
            try:
                # x = tfd.MultivariateNormalFullCovariance(S @ (Sigma_inv @ As[state] @ x + Thetas[t]), S).sample()
                x = multivariate_normal.rvs(mean=S @ (Sigma_inv @ As[state] @ x + Thetas[t]), cov=S)
            except np.linalg.LinAlgError:
                S = turn_matrix_positive_semidefinite(S)
                try:
                    x = multivariate_normal.rvs(mean=S @ (Sigma_inv @ As[state] @ x + Thetas[t]), cov=S)
                except:
                    S = aggresively_turn_matrix_positive_semidefinite(S)
                    x = multivariate_normal.rvs(mean=S @ (Sigma_inv @ As[state] @ x + Thetas[t]), cov=S)
            res[t] = x
        return res[:, :self.dynamics.x_dim]

    def _random_state_init(self, L):
        res = np.zeros(L)
        z, d = self.initial.sample()
        for i in range(L):
            res[i] = z
            if d > 1:
                d -= 1
            else:
                z = self.transition.sample_x(z)
                d = self.duration.sample_d(z)
        return res

    def _expand_Z(self, Z):
        return [np.array([x[0] for x in z]) for z in Z]

    def _expand_D(self, Z):
        return [np.array([x[1] for x in z]) for z in Z]

    # @profile
    def _plot_everything(self, path, name, iter, X, Y, Y_est, Z, As, Dseq, states_n, log_to_wandb, save_space = True):
        if not save_space:
            name = f"{name}_{iter}"
        plot_dict = {}
        file_path = os.path.join(path, f"{name}_dynamics_actual_shift.png")
        plot_actual_shifts(X, Z, states_n, file_path, title=f"{name} - as - {iter}")
        plot_dict["dynamics_actual_shift"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_estimated_dynamics_time.png")
        plot_observations(Z[:1000], X[:1000], fname=file_path, title=f"{name} - estimated_dynamics_time - {iter}")
        plot_dict["estimated_dynamics_time"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_phase_portrait.png")
        if self.dynamics.affine:
            plot_phase_portrait(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), Z, As, states_n, file_path, title=f"{name} - pp - {iter}")
        else:
            plot_phase_portrait(X, Z, As, states_n, file_path, title=f"{name} - pp - {iter}")
        plot_dict["phase_portrait"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_emissions.png")
        plot_emission(Y, Z, states_n, file_path, title=f"{name} - emission - {iter}")
        plot_dict["emissions"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_estimated_emissions.png")
        plot_emission(Y_est, Z, states_n, file_path, title=f"{name} - estimated_emission - {iter}")
        plot_dict["estimated_emissions"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_estimated_emissions_time.png")
        plot_observations(Z[:1000], Y_est[:1000], fname=file_path, title=f"{name} - estimated_emission_time - {iter}")
        plot_dict["estimated_emissions_time"] = wandb.Image(file_path)
        file_path = os.path.join(path, f"{name}_durations.png")
        fig, ax = plot_durations(Z, Dseq, states_n, title=f"{name} - durations - {iter}")
        fig.savefig(file_path)
        plt.close(fig)
        plot_dict["durations"] = wandb.Image(file_path)
        if log_to_wandb:
            wandb.log(plot_dict, step=iter)
        plt.close('all')

    def _kalman_generator(self, Y, Z):
        def _kalman_pass(i):
            y = Y[i]
            Lambdas, Thetas = self._backward_kalman_filter(y, Z[i])
            Xi = self._forward_kalman_sample(Lambdas, Thetas, Z[i])
            return Xi

        return _kalman_pass

    def _alphas_generator(self, X, Y, U, iter, init_iters, decay):
        def _alphas_pass(i):
            Xi = X[i]
            Yi = Y[i]
            X_priors = [self.dynamics.sample_obs(k) for k in self.states]
            return self.beam_forward(X_priors=X_priors, X=Xi, Y=Yi, U=U[i], iter=iter, init_iters=init_iters, decay=decay)

        return _alphas_pass

    def _Zs_generator(self, alphas, U, decay):
        def _alphas_pass(i):
            return self.beam_backward_sample(alphas[i], U[i], decay=decay)

        return _alphas_pass

    def _beam(self, name, online, count, min_d, max_d, sample_U, Z_samples, min_u, log_to_wandb, X, Y, decay, update_D,
              plot, plot_folder, prev_X, prev_Y_est, burnin, dump_period, dump_path, num_of_workers, actual_Z, cache,
              double_sample=False, init_iters=100, save_space=True):
        # print(h.heap())
        # print(h.heap().byid[0].sp)
        log.info('\n\nrunning sample %s' % count)

        log.debug('getting support')
        self.set_duration_support(min_d, max_d, cache=cache)
        # self.set_transition_loglikelihood()

        # slice
        start = time.time()
        if sample_U:
            U = []
            for Z in Z_samples:
                U.append(self.slice_sample(Z, min_u=min_u))
        else:
            U = []
            for Z in Z_samples:
                U.append(np.zeros(len(Z)))

        log.debug('slice sample took %ss' % (time.time() - start))
        # states
        start = time.time()
        if online:
            with Pool(num_of_workers, len(X)) as pool:
                alphas = pool.map(self._alphas_generator(X, Y, U, count, init_iters, decay), range(len(X)))
                Z_samples = pool.map(self._Zs_generator(alphas, U, decay), range(len(X)))
            log.debug('inference took %ss' % (time.time() - start))
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

        if prev_X is not None and plot:
            plot_dict = {}
            try:
                file_path = os.path.join(plot_folder, f"{name}_{count}_dynamics_shift.png")

                plot_distribution_shift(prev_X[0], X[0], file_path, name_X="previous", name_Y="current",
                                        title=f"{name} - ds - {count}")
                plot_dict["dynamics_shift"] = wandb.Image(file_path)
            except Exception as e:
                print("DS plot failed")
                print(e)
            try:
                file_path = os.path.join(plot_folder, f"{name}_{count}_emissions_shift.png")
                plot_distribution_shift(prev_Y_est[0], Y_est[0], file_path, name_X="previous", name_Y="current",
                                        title=f"{name} - es - {count}")
                plot_dict["emissions_shift"] = wandb.Image(file_path)
            except Exception as e:
                print("DS plot failed")
                print(e)
            if log_to_wandb != "":
                wandb.log(plot_dict, step=count)
        prev_X = X
        prev_Y_est = Y_est
        X = []

        # if wandb_log != "":
        #     log_wandb_scalar_or_array(np.concatenate(Z_samples_expanded), "Z_samples_expanded", step=count)
        with Pool(num_of_workers, len(Y)) as pool:
            X = pool.map(self._kalman_generator(Y, Z_samples_expanded), range(len(Y)))
            if double_sample:
                Z_samples = pool.map(self._Zs_generator(alphas, U, decay), range(len(X)))
                Z_samples_expanded = self._expand_Z(Z_samples)
                D_samples_expanded = self._expand_D(Z_samples)
            pool.clear()
        # for i, y in enumerate(Y):
        #     Lambdas, Thetas = self._backward_kalman_filter(y, Z_samples_expanded[i])
        #     Xi = self._forward_kalman_sample(Lambdas, Thetas, Z_samples_expanded[i])
        #     X.append(Xi)
        if update_D:
            self.duration.update(Z_samples)

        self.transition.update(Z_samples)
        # loglikelihood
        self.dynamics.update(Xs=X, Zs=Z_samples_expanded)
        self.emission.update(Zs=Z_samples_expanded, Xs=X, Ys=Y)
        l = self.loglikelihood(Z_samples, X, Y)
        if log_to_wandb:
            self.duration.log_to_wandb(count)
            self.transition.log_to_wandb(count)
            self.dynamics.log_to_wandb(count)
            self.emission.log_to_wandb(count)
            wandb.log({"loglikelihood": l}, step=count)
        # L.append(l)
        log.info("log loglikelihood at iteration %s: %s" % (count, l))

        if count > burnin:
            if count % dump_period == 0:
                log.debug('writing iteration %s to disk' % count)
                self.dump(name, dump_path)
        return prev_X, X, Z_samples, prev_Y_est

    # @profile
    def beam(self, Y, min_u=0, its=100, burnin=50, name='beamer', online=True, sample_U=True, update_D=True,
             force_U=None, min_d=None, max_d=None, wandb_log="", decay=None, plot=False, plot_folder="./Plots/",
             dump_period=5, dump_path="./", num_of_workers=5, actual_Z=None, double_sample=False, cache=None,
             init_with_pca=True, initial_states = None, init_iters=100, complex_init=False,fast = False, return_accuracy=False):
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
        init_with_pca = init_with_pca and self.dynamics.x_dim <= self.emission.obs_dim
        actual_Z_concatenated = None if actual_Z is None else np.concatenate(actual_Z)
        # get support of duration distributions
        self.set_duration_support(min_d, max_d, cache=cache)
        # self.set_transition_loglikelihood()

        # sample auxillary variables from some small value
        if force_U:
            U = force_U
        else:
            U = [np.random.uniform(min_u, 0.000000000001, size=len(Yi)) for Yi in Y]

        # get worthy samples given the relaxed U
        alphas = []
        if initial_states is None:
            Z_samples = [self._random_state_init(len(y)) for y in Y]
        else:
            Z_samples = initial_states
        X = []
        for i, y in enumerate(Y):
            Xi: np.ndarray
            if init_with_pca and not initial_states:
                pca = PCA(n_components=self.dynamics.x_dim)
                Xi = pca.fit_transform(y)
            else:
                Lambdas, Thetas = self._backward_kalman_filter(y, Z_samples[i])
                Xi = self._forward_kalman_sample(Lambdas, Thetas, Z_samples[i])
                if complex_init:
                    with Pool(num_of_workers, len(Y)) as pool:
                        alphas = pool.map(self._alphas_generator(X, Y, U, 0, init_iters, decay), range(len(X)))
                        Z_samples = pool.map(self._Zs_generator(alphas, U, decay), range(len(X)))
            X.append(Xi)
        if init_with_pca or initial_states:
            # Z_samples_expanded = self._expand_Z(Z_samples)
            self.dynamics.update(Xs=X, Zs=Z_samples)
            self.emission.update(Zs=Z_samples, Xs=X, Ys=Y)
            self.duration.update(Z=Z_samples, z_tuples=False)
        # Dotąd git

        log.debug('performing inference')
        Z_samples = []
        if online:
            for i, Xi in enumerate(X):
                Yi = Y[i]
                X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                alphas.append(self.beam_forward(X_priors=X_priors, X=Xi, Y=Yi, U=U[i], iter=0, init_iters=init_iters, decay=decay))
                # GITES_CHECK - chyba git
                Z_samples.append(self.beam_backward_sample(alphas[i], U[i], decay=decay))
        else:
            raise NotImplementedError
        # count how many iterations we've done so far
        count = 0

        # block gibbs
        prev_X = None
        prev_Y_est = None
        res_accuracies = []
        acc_its = its - 10

        for count in range(its):
            log_to_wandb = wandb_log != "" and ((count % dump_period == 0) or (count == its - 1))
            prev_X, X, Z_samples, prev_Y_est = self._beam(name, online, count, min_d, max_d, sample_U, Z_samples, min_u,
                                                          log_to_wandb, X, Y, decay, update_D, plot=plot,
                                                          plot_folder=plot_folder, prev_X=prev_X, prev_Y_est=prev_Y_est,
                                                          burnin=burnin, dump_period=dump_period, dump_path=dump_path,
                                                          num_of_workers=num_of_workers, actual_Z=actual_Z, cache=cache,
                                                          double_sample=double_sample, init_iters=init_iters)
            if actual_Z is not None and log_to_wandb:
                Z = np.concatenate(self._expand_Z(Z_samples))
                try:
                    Z = permute(Z, actual_Z_concatenated, self.L)
                except:
                    print("permutation failed")
                log_wandb_scalar_or_array(accuracy_score(Z, actual_Z_concatenated), 'accuracy', step=count)
                log_wandb_scalar_or_array(confusion_matrix(Z, actual_Z_concatenated), 'confusion_matrix', step=count)
                log_wandb_scalar_or_array(confusion_matrix(Z, actual_Z_concatenated, normalize='true'),
                                          'confusion_matrix_normalized', step=count)
                if not plot:
                    continue
                Z = self._expand_Z([Z_samples[0]])[0]
                Z = permute(Z, actual_Z[0], self.L)
                file_path = os.path.join(plot_folder, f"{name}_states.png")
                plot_states(data_z=actual_Z[0][:1000], z_est=Z[:1000], label="states", fname=file_path)
                wandb.log({"estimated_states": wandb.Image(file_path)}, step=count)
                if count >= acc_its:
                    res_accuracies.append(accuracy_score(Z, actual_Z_concatenated))
        if return_accuracy:
            return Z_samples, X, np.mean(np.array(res_accuracies))
        return Z_samples, X

    # @profile
    def beam_forward(self, X_priors, X, Y, U, iter, init_iters, W=None, decay=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """

        log.info('running forward algorithm')

        # initialise alphahat
        alphahat = [{} for _ in X]

        log.debug('calculating observation loglikelihoods')
        T = len(X)
        X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
        Y_arr = np.array(Y) if not isinstance(Y, np.ndarray) else Y

        if iter >= init_iters and hasattr(self.dynamics, 'loglikelihood_batch') and hasattr(self.emission, 'loglikelihood_batch'):
            # Batch computation for efficiency
            # Build prev_X: prev_X[0] uses X_priors mean, prev_X[t] = X[t-1] for t > 0
            prev_X = np.zeros_like(X_arr)
            prev_X[0] = np.mean([X_priors[i] for i in self.states], axis=0)
            prev_X[1:] = X_arr[:-1]

            ol_dyn = self.dynamics.loglikelihood_batch(prev_X, X_arr)  # (K, T)
            ol_emi = self.emission.loglikelihood_batch(X_arr, Y_arr)  # (K, T)
            ol = ol_dyn + ol_emi
            ol = np.maximum(-1000000000000, ol)
        else:
            # Original loop for initial iterations or if batch methods not available
            ol = np.zeros((self.K, T))
            for i in self.states:
                prev_x = X_priors[i]
                for t, y in enumerate(Y):
                    x = X[t]
                    ol[i, t] = self.dynamics.loglikelihood(state=i, observation=x,
                                                           previous_observation=prev_x) + self.emission.loglikelihood(
                        state=i, observation=y, dynamics=x)
                    ol[i, t] = np.maximum(-1000000000000, ol[i, t])
                    if iter >= init_iters:
                        prev_x = x
        log.debug('starting iteration')

        worthy_time = 0
        alpha_time = 0

        # GITES_CHECK
        for t, y in enumerate(Y):
            start = time.time()
            if W is None:
                if t == 0:
                    worthy = self.get_initial_worthy(U[t])
                else:
                    worthy = self.get_worthy(U[t], worthy, decay)
            else:
                worthy = W[t]
            worthy_time += time.time() - start

            start = time.time()
            if t == 0:
                for i in self.states:
                    alphahat[t][i] = {}
                    for d in [1] + list(range(self.left[i], self.right[i] + 1)):
                        alphahat[t][i][d] = self.initial.loglikelihood((i, d))

            else:
                for i, J in worthy.items():
                    # initialise alpahat[t] if necessary
                    if i[0] not in alphahat[t]:
                        alphahat[t][i[0]] = {i[1]: -1000000000000}
                    else:
                        if i[1] not in alphahat[t][i[0]]:
                            alphahat[t][i[0]][i[1]] = -1000000000000

                    # here i is those (state,duration)s worth figuring out for
                    # alpha hat. Then J is a list of those indices into the
                    # previous alpha hat we should sum over to find the next
                    # alpha hat.

                    # so you can read this indexing as
                    # alphahat[time][state][duration]

                    for j in J:
                        try:
                            alphahat[t][i[0]][i[1]] = elnsum(
                                alphahat[t][i[0]][i[1]],
                                alphahat[t - 1][j[0]][j[1]]
                            )
                        except KeyError:
                            # if a KeyError occurred, then we already decided
                            # that alphahat[t-1][state][duration] was zero, so
                            # we can just ignore it
                            # print "skipping over a key error"
                            pass

                    alphahat[t][i[0]][i[1]] += ol[i[0], t]
                    assert not np.isinf(alphahat[t][i[0]][i[1]])

            try:
                assert alphahat[t], "alpha[%s]:%s" % (t, alphahat[t])
            except AssertionError:
                print("alpha[%s]:%s" % (t - 1, alphahat[t - 1]))
                print(worthy)
                raise
            alpha_time += time.time() - start

        log.debug('time spent building alpha: %s' % alpha_time)
        log.debug('time spent finding worthy: %s' % worthy_time)
        return alphahat

    # @profile
    def set_duration_support(self, min_d, max_d, cache=None):
        if (max_d == None) and (min_d == None):
            self.left, self.right = zip(*[self.duration.support(i, cache=cache) for i in self.states])
        else:
            self.right = max_d
            self.left = min_d

    def slice_sample(self, Z, min_u=0):
        log.info('forming slice')
        u = [np.array(min_u)]

        for t in range(1, len(Z)):
            i = Z[t - 1][0]
            j = Z[t][0]
            di = Z[t - 1][1]
            dj = Z[t][1]
            try:
                u.append(
                    np.random.uniform(
                        low=min_u,
                        high=self._get_likelihood(i, j, di, dj))
                )
            except KeyError:
                raise
        return np.array(u)

    def _infer(self, online, count, min_d, max_d, sample_U, Z_samples, min_u, X, Y, decay, update_D, num_of_workers,
               cache):
        log.info('\n\nrunning sample %s' % count)

        log.debug('getting support')
        self.set_duration_support(min_d, max_d, cache=cache)
        # self.set_transition_loglikelihood()

        # slice
        start = time.time()
        if sample_U:
            U = []
            for Z in Z_samples:
                U.append(self.slice_sample(Z, min_u=min_u))

        log.debug('slice sample took %ss' % (time.time() - start))

        # states
        start = time.time()
        # alphas = []
        # Z_samples = []
        if online:
            with Pool(num_of_workers, len(X)) as pool:
                alphas = pool.map(self._alphas_generator(X, Y, U, 0, 0, decay), range(len(X)))
                Z_samples = pool.map(self._Zs_generator(alphas, U, decay), range(len(X)))
            log.debug('inference took %ss' % (time.time() - start))
        else:
            raise NotImplementedError

        Z_samples_expanded = self._expand_Z(Z_samples)

        # if wandb_log != "":
        #     log_wandb_scalar_or_array(np.concatenate(Z_samples_expanded), "Z_samples_expanded", step=count)
        with Pool(num_of_workers, len(Y)) as pool:
            X = pool.map(self._kalman_generator(Y, Z_samples_expanded), range(len(Y)))

        return X, Z_samples

    # @profile
    def infer(self, Y, min_u=0, its=10, online=True, sample_U=True, update_D=True,
              force_U=None, min_d=None, max_d=None, decay=None, num_of_workers=5, actual_Z=None, cache=None):
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
        # get support of duration distributions
        self.set_duration_support(min_d, max_d, cache=cache)
        # self.set_transition_loglikelihood()

        # sample auxillary variables from some small value
        if force_U:
            U = force_U
        else:
            U = [np.random.uniform(min_u, 0.000000000001, size=len(Yi)) for Yi in Y]

        # get worthy samples given the relaxed U
        alphas = []
        Z_samples = [self._random_state_init(len(y)) for y in Y]

        X = []
        for i, y in enumerate(Y):
            Lambdas, Thetas = self._backward_kalman_filter(y, Z_samples[i])
            Xi = self._forward_kalman_sample(Lambdas, Thetas, Z_samples[i])
            X.append(Xi)

        log.debug('performing inference')
        Z_samples = []
        if online:
            for i, Xi in enumerate(X):
                Yi = Y[i]
                X_priors = [self.dynamics.sample_obs(k) for k in self.states]
                alphas.append(self.beam_forward(X_priors=X_priors, X=Xi, Y=Yi, U=U[i], iter=0, init_iters=0, decay=decay))
                # GITES_CHECK - chyba git
                Z_samples.append(self.beam_backward_sample(alphas[i], U[i], decay=decay))
        else:
            raise NotImplementedError
        # count how many iterations we've done so far
        for count in range(its):
            X, Z_samples = self._infer(online=online, count=count, min_d=min_d, max_d=max_d, sample_U=sample_U,
                                       Z_samples=Z_samples, min_u=min_u, X=X, Y=Y, decay=decay, update_D=update_D,
                                       num_of_workers=num_of_workers, cache=cache)
        return Z_samples, X

    def get_worthy(self, u, old_worthy, decay=None):
        """
        decides which transitions are valid given the auxilliary variable
        """
        worthy = {}
        # we only consider those transitions that are possible from
        # t-1
        for i, di in old_worthy:
            # which transitions are worthy?
            if di == 1:
                for j in self.states:
                    for dj in range(1, self.right[i] + 1):
                        # if the probability is worthy..
                        if self._get_likelihood(i, j, di, dj) > u:
                            # add it to the list!
                            try:
                                worthy[(j, dj)].append((i, di))
                            except KeyError:
                                # (or start a new list)
                                worthy[(j, dj)] = [(i, di)]
            else:
                j = i
                dj = di - 1
                try:
                    worthy[(j, dj)].append((i, di))
                except KeyError:
                    worthy[(j, dj)] = [(i, di)]
        if len(worthy) == 0 and decay is not None:
            worthy = self.get_worthy(decay * u, old_worthy, decay)
        assert worthy, (u, old_worthy)
        return worthy

    def get_initial_worthy(self, u):
        """
        gets the intial set of worthy states
        """
        worthy = {}
        for i in self.states:
            for di in range(self.left[i], self.right[i] + 1):
                for j in self.states:
                    for dj in range(self.left[j], self.right[j] + 1):
                        if self._get_likelihood(i, j, di, dj) > u:
                            try:
                                worthy[(j, dj)].append((i, di))
                            except KeyError:
                                worthy[(j, dj)] = [(i, di)]
        return worthy

    # @profile
    def beam_backward_sample(self, alphahat, U, W=None, decay=None):
        """
        perfomrs the backwards sweep given the forwards sweep and the auxilliary variables
        """

        log.info('backward sampling state sequence')

        def sample_z(a):
            vals = []
            # Added by Mikolaj Slupinsi
            a_keys = list(a.keys())
            for k in a_keys:
                if len(a[k]) == 0:
                    del a[k]
            for i in a:
                vals.extend(a[i].values())
            try:
                m = np.array(vals).max()
            except ValueError:
                print(a)
                print(vals)
                raise
            p = [np.exp(np.array(list(a[i].values())) - m).sum() for i in a.keys()]

            xi = Categorical(np.array(p)).sample()
            x = list(a.keys())[xi]
            try:
                p = np.exp(np.array(list(a[x].values())) - max(list(a[x].values())))
                di = Categorical(p).sample()
            except AssertionError:
                print(p)
                raise
            d = list(a[x].keys())[di]
            return x, d

        T = len(alphahat)
        try:
            Z = [sample_z(alphahat[-1])]
        except ValueError:
            print(alphahat[-1])
            raise
        for t in reversed(range(T - 1)):
            # pick the subset of alphahats
            # here w[t+1][Z[-1]] is a list of the possible zs you can sample
            # from in alphahat[t] given that the next state is Z[-1], i.e.
            # w[t+1][Z[t+1]] is the next state

            # a = dict([(i,{}) for i in self.states])
            # for j in worthy[Z[-1]]:
            #    try:
            #        a[j[0]][j[1]] = alphahat[t][j[0]][j[1]]
            #    except KeyError:
            #        a[j[0]][j[1]] = 0

            # we need to build up a pair of worthys

            # first, the get_worthy method uses old_worthy to make sure that
            # the transitions are consistent. So we need just the keys in
            # alphahat as we know that this is 'old worthy' for the worthy
            # variables at t+1
            old_worthy = {}
            for state in alphahat[t]:  # I'm not sure there is no off by one here
                for duration in alphahat[t][state]:
                    key = (state, duration)
                    old_worthy[key] = 0

            worthy = self.get_worthy(U[t + 1], old_worthy, decay=decay)

            a = dict([(i, {}) for i in self.states])
            try:
                worthy[Z[-1]]
            except KeyError:
                print(worthy)
                raise

            for j in worthy[Z[-1]]:
                try:
                    a[j[0]][j[1]] = alphahat[t][j[0]][j[1]]
                except KeyError:
                    a[j[0]][j[1]] = -10000000
            z = sample_z(a)
            Z.append(z)
        Z.reverse()
        return Z

    def _get_likelihood(self, i, j, di, dj):
        return np.exp(self._get_loglikelihood(i, j, di, dj))

    def _get_loglikelihood(self, i, j, di, dj):
        if di == 1:
            return self.transition.loglikelihood(i, j) + self.duration.loglikelihood(j, dj)
        elif i == j and dj == di - 1:
            return 0
        return -1000000000000

    def loglikelihood(self, Zs, Xs, Ys):
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
        l = 0
        for Z, X, Y in zip(Zs, Xs, Ys):
            for t in range(1, len(Z)):
                i = Z[t - 1][0]
                j = Z[t][0]
                di = Z[t - 1][1]
                y = Y[t]
                x = X[t]
                prev_x = X[t - 1]
                if i == j:
                    l += self.emission.loglikelihood(j, x, y)
                    l += self.dynamics.loglikelihood(j, prev_x, x)
                else:
                    l += (
                            self.transition.loglikelihood(i, j) +
                            self.duration.loglikelihood(i, di) +
                            self.emission.loglikelihood(j, x, y) +
                            self.dynamics.loglikelihood(j, prev_x, x))
        return l

    # @profile
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
            prev_x = None
            d = self.duration.sample_d(z)
        else:
            z, d, prev_x = init
            if d > 1:
                d -= 1
            else:
                z = self.transition.sample_x(z)
                d = self.duration.sample_d(z)
        for t in range(T):
            x = self.dynamics.sample_obs(z, prev_x)
            y = self.emission.sample_obs(z, x)
            yield z, x, y, d
            if d > 1:
                d -= 1
            else:
                z = self.transition.sample_x(z)
                d = self.duration.sample_d(z)
            prev_x = x

    # @profile
    def sim(self, T, init=None):
        """
        Draws a sequence of length T from the EDHMM

        Parameters
        ----------
        T : int
            number of time points
        """
        Z, X, Y, D = [], [], [], []
        for z, x, y, d in self.gen(T, init):
            Z.append(z)
            X.append(x)
            Y.append(y)
            D.append(d)
        return Z, X, Y, D
