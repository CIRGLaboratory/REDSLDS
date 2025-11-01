import logging
import pprint

from .duration import DummyDuration
from .dynamics import AbstractDynamics
from .emission import AbstractEmission
from .initial import LearnableInitial
from .redslds import REDSLDS
from .transition import RecurrentTransition

pp = pprint.PrettyPrinter(indent=4)

log = logging.getLogger('rslds')


class RSLDS(REDSLDS):
    def __init__(self, K: int, emission: AbstractEmission, dynamics: AbstractDynamics, nobeam=True,
                 extended_kalman=False, affine=False, eps=10000.0, nonswitching=False,
                 forward_discrete_states_sample=False):
        transition = RecurrentTransition(K, D=dynamics.get_D(), loopy=True, affine=affine, eps=eps,
                                         nonswitching=nonswitching)
        duration = DummyDuration(K=K, D=dynamics.get_D(), recurrent=True)
        initial = LearnableInitial(K=K, dur_dist=duration, dynamics_dist=dynamics)
        super().__init__(initial=initial, transition=transition, emission=emission, dynamics=dynamics,
                         duration=duration, nobeam=nobeam, extended_kalman=extended_kalman,
                         forward_discrete_states_sample=forward_discrete_states_sample)
