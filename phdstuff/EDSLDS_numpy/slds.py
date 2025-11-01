import logging
import numpy as np
import pprint

from .duration import TestPoisson, DummyDuration
from .dynamics import LinearGaussianDynamics, AbstractDynamics
from .edslds import EDSLDS
from .emission import LinearGaussianEmission, AbstractEmission
from .initial import Initial, LearnableInitial, AbstractInitial
from .transition import Transition, LoopyTransition

pp = pprint.PrettyPrinter(indent=4)

log = logging.getLogger('slds')


class NoBeamEDSLDS(EDSLDS):
    def __init__(self, initial: AbstractInitial, transition: Transition, emission: AbstractEmission,
                 dynamics: AbstractDynamics, duration: TestPoisson):
        super().__init__(initial, transition, emission, dynamics, duration)

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
        return NoBeamEDSLDS(initial, transition, emission, dynamics, duration)

    def _alphas_generator(self, X, Y, U, iter, init_iters, decay):
        U = [np.zeros_like(u) for u in U]
        return super(NoBeamEDSLDS, self)._alphas_generator(X=X, Y=Y, U=U, iter=iter, init_iters=init_iters, decay=decay)

    def _Zs_generator(self, alphas, U, decay):
        U = [np.zeros_like(u) for u in U]
        return super(NoBeamEDSLDS, self)._Zs_generator(alphas=alphas, U=U, decay=decay)

    def _beam(self, name, online, count, min_d, max_d, sample_U, Z_samples, min_u, log_to_wandb, X, Y, decay, update_D,
              plot, plot_folder, prev_X, prev_Y_est, burnin, dump_period, dump_path, num_of_workers, actual_Z, cache,
              double_sample=False, init_iters=100, save_space=True):
        _ = sample_U
        return super(NoBeamEDSLDS, self)._beam(name=name, online=online, count=count, min_d=min_d, max_d=max_d,
                                               sample_U=False, Z_samples=Z_samples, min_u=0., log_to_wandb=log_to_wandb,
                                               X=X, Y=Y, decay=decay, update_D=update_D,
                                               plot=plot, plot_folder=plot_folder, prev_X=prev_X, prev_Y_est=prev_Y_est,
                                               burnin=burnin, dump_period=dump_period, dump_path=dump_path,
                                               num_of_workers=num_of_workers, actual_Z=actual_Z, cache=cache,
                                               double_sample=False, init_iters=init_iters, save_space=save_space)

    # @profile
    def beam(self, Y, min_u=0, its=100, burnin=50, name='noslicing', online=True, sample_U=True, update_D=True,
             force_U=None, min_d=None, max_d=None, wandb_log="", decay=None, plot=False, plot_folder="./Plots/",
             dump_period=5, dump_path="./", num_of_workers=5, actual_Z=None, double_sample=False, cache=None,
             init_with_pca=True, initial_states=None, init_iters=100, complex_init=False, save_space=True):
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
        _ = sample_U
        force_U = [np.zeros(len(Yi)) for Yi in Y]
        return super(NoBeamEDSLDS, self).beam(Y=Y, min_u=min_u, its=its, burnin=burnin, name=name, online=online,
                                              sample_U=False, update_D=update_D,
                                              force_U=force_U, min_d=min_d, max_d=max_d, wandb_log=wandb_log,
                                              decay=decay, plot=plot, plot_folder=plot_folder,
                                              dump_period=dump_period, dump_path=dump_path,
                                              num_of_workers=num_of_workers, actual_Z=actual_Z,
                                              double_sample=double_sample, cache=cache,
                                              init_with_pca=init_with_pca, initial_states=initial_states,
                                              init_iters=init_iters, complex_init=complex_init, save_space=save_space)

    # @profile
    def beam_forward(self, X_priors, X, Y, U, iter, init_iters, W=None, decay=None):
        """
        runs the forward algorithm, sampling only from valid transitions
        """
        return super(NoBeamEDSLDS, self).beam_forward(X_priors=X_priors, X=X, Y=Y, U=np.zeros_like(U), iter=iter,
                                                      init_iters=init_iters, W=W, decay=decay)

    def slice_sample(self, Z, min_u=0):
        log.info('forming slice')
        return np.zeros(len(Z))

    def _infer(self, online, count, min_d, max_d, sample_U, Z_samples, min_u, X, Y, decay, update_D, num_of_workers,
               cache):
        _ = sample_U
        return super(NoBeamEDSLDS, self)._infer(online=online, count=count, min_d=min_d, max_d=max_d, sample_U=False,
                                                Z_samples=Z_samples, min_u=0., X=X, Y=Y, decay=decay, update_D=update_D,
                                                num_of_workers=num_of_workers,
                                                cache=cache)

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
        _ = sample_U
        force_U = [np.zeros(len(Yi)) for Yi in Y]
        return super(NoBeamEDSLDS, self).infer(Y=Y, min_u=0, its=its, online=online, sample_U=False, update_D=update_D,
                                               force_U=force_U, min_d=min_d, max_d=max_d, decay=decay,
                                               num_of_workers=num_of_workers, actual_Z=actual_Z, cache=cache)

    def get_worthy(self, u, old_worthy, decay=None):
        """
        decides which transitions are valid given the auxilliary variable
        """
        return super(NoBeamEDSLDS, self).get_worthy(0., old_worthy=old_worthy, decay=decay)

    def get_initial_worthy(self, u):
        """
        gets the intial set of worthy states
        """
        return super(NoBeamEDSLDS, self).get_initial_worthy(0.)

    # @profile
    def beam_backward_sample(self, alphahat, U, W=None, decay=None):
        """
        perfomrs the backwards sweep given the forwards sweep and the auxilliary variables
        """
        return super(NoBeamEDSLDS, self).beam_backward_sample(alphahat=alphahat, U=np.zeros_like(U), W=W, decay=decay)


class SLDS(NoBeamEDSLDS):
    def __init__(self, K: int, emission: AbstractEmission, dynamics: AbstractDynamics):
        transition = LoopyTransition(K, D=dynamics.get_D())
        duration = DummyDuration(K=K, D=dynamics.get_D(), recurrent=True)
        initial = LearnableInitial(K=K, dur_dist=duration, dynamics_dist=dynamics)
        super().__init__(initial, transition, emission, dynamics, duration)
