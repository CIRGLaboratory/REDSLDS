import logging
import pprint


from .duration import DummyDuration
from .dynamics import AbstractDynamics
from .redslds import REDSLDS
from .emission import AbstractEmission
from .initial import LearnableInitial
from .transition import LoopyTransition

pp = pprint.PrettyPrinter(indent=4)

log = logging.getLogger('sldsv2')


class SLDSv2(REDSLDS):
    def __init__(self, K: int, emission: AbstractEmission, dynamics: AbstractDynamics, nobeam=True,
                 extended_kalman=True, forward_discrete_states_sample=False):
        transition = LoopyTransition(K, D=dynamics.get_D())
        duration = DummyDuration(K=K, D=dynamics.get_D(), recurrent=True)
        initial = LearnableInitial(K=K, dur_dist=duration, dynamics_dist=dynamics)
        super().__init__(initial=initial, transition=transition, emission=emission, dynamics=dynamics,
                         duration=duration, nobeam=nobeam, extended_kalman=extended_kalman,
                         forward_discrete_states_sample=forward_discrete_states_sample)
