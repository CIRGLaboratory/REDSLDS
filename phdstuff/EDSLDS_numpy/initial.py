import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional

from .duration import RecurrentDuration, DummyDuration
from .dynamics import AbstractDynamics

log = logging.getLogger('initial')
from scipy.stats import expon, multinomial

from .utils import *


class AbstractInitial(ABC):
    @abstractmethod
    def dump(self, path, name):
        pass

    @staticmethod
    def load(path, name):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def loglikelihood(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class Categorical:
    """
    Defines a Categorical Distribution
    """

    def __init__(self, p):
        assert all(p >= 0), p
        assert not any(np.isinf(p))
        p += 0.000000001
        assert p.sum()
        self.p = p / p.sum()
        assert self.p.sum().round(5) == 1, (p, self.p, self.p.sum())
        self.p = np.squeeze(self.p)

    def sample(self):
        if self.p.shape == ():
            return 0
        try:
            # x = tfd.Multinomial(1, self.p).sample()
            x = multinomial.rvs(1, self.p)
        except ValueError:
            # print self.p
            # print self.p.sum()
            raise
        except TypeError:
            print(self.p)
            print(type(self.p))
            print(self.p.shape)
            raise
        return int(np.where(x == 1)[0][0])

    def likelihood(self, k):
        if k < 0 or k >= len(self.p):
            return 0.
        return self.p[k]

    def log_likelihood(self, k):
        return np.log(self.likelihood(k))

    def log_prob(self, k):
        return self.log_likelihood(k)

    def dump(self, path, name):
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_cat.pickle"), "wb") as ofile:
            pickle.dump(self.p, ofile)

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_cat.pickle"), "rb") as ifile:
            p = pickle.load(ifile)
        return Categorical(p)


class Initial(AbstractInitial):
    """
    Defines an Initial distribution
    """

    def __init__(self, K, beta=0.001):
        self.K = K
        self.beta = beta
        # self.state_dist = tfd.Categorical(probs=[1. / K for _ in range(K)], name="StateInit")
        self.state_dist = Categorical(np.array([1. / K for _ in range(K)]))
        # self.dur_dist = tfd.Exponential(beta, name="DurInit")
        self.dur_dist = expon

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
            "beta": self.beta
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_init.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)
        self.state_dist.dump(path, f"{name}_init")

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, f"{name}_init.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        res = Initial(**kwargs)
        res.state_dist = Categorical.load(path, f"{name}_init")
        return res

    def __call__(self, z=None):
        if z is None:
            return self.sample()
        else:
            return self.likelihood(z)

    def __len__(self):
        return self.K

    def sample(self):
        # self.dist.draw_from_prior()
        x = self.state_dist.sample()
        d = self.dur_dist.rvs(scale=1. / self.beta)
        # x = int(self.dist.s_init.value)
        # d = int(round(self.dist.d_init.value))
        return int(x), int(d)

    def loglikelihood(self, z):

        assert z[0] in range(self.K)
        assert z[1] > 0, z
        #
        # self.dist.s_init.set_value(z[0])
        # self.dist.d_init.set_value(z[1])
        # l_x = self.dist.s_init.logp
        # l_d = self.dist.d_init.logp
        l_x = self.state_dist.log_prob(z[0])
        l_d = self.dur_dist.logpdf(z[1])
        return l_x + l_d

    def update(self, _E):
        raise NotImplementedError

    def report(self):
        report = "initial distribution: K=%s, beta=%s" % (self.K, self.beta)
        log.info(report)


class LearnableInitial(AbstractInitial):
    """
    Defines an Initial distribution
    """

    def __init__(self, K, dur_dist, dynamics_dist: Optional[AbstractDynamics] = None, input_dim=None):
        self.K = K
        # self.state_dist = tfd.Categorical(probs=[1. / K for _ in range(K)], name="StateInit")
        self.state_dist = Categorical(np.array([1. / K for _ in range(K)]))
        # self.dur_dist = tfd.Exponential(beta, name="DurInit")
        self.dur_dist = dur_dist
        self.dynamics_dist = dynamics_dist
        self.input_dim = input_dim

    def dump(self, path, name):
        dump_dic = {
            "K": self.K,
        }
        with open(os.path.join(path, f"{self.__class__.__name__}_{name}_init.pickle"), "wb") as ofile:
            pickle.dump(dump_dic, ofile)
        self.state_dist.dump(path, f"{name}_init")

    @staticmethod
    def load(path, name, dur_dist):
        with open(os.path.join(path, f"{name}_init.pickle"), "rb") as ifile:
            kwargs = pickle.load(ifile)
        res = LearnableInitial(K=kwargs["K"], dur_dist=dur_dist)
        res.state_dist = Categorical.load(path, f"{name}_init")
        return res

    def __call__(self, z=None):
        if z is None:
            return self.sample()
        else:
            return self.likelihood(z)

    def __len__(self):
        return self.K

    def sample(self):
        # self.dist.draw_from_prior()
        s = self.state_dist.sample()
        if isinstance(self.dur_dist, RecurrentDuration) or (
                isinstance(self.dur_dist, DummyDuration) and self.dur_dist.recurrent):
            x0 = self.dynamics_dist.sample_obs(s)
            d = self.dur_dist.sample_d(s, x0)
            return int(s), int(d), x0
        d = self.dur_dist.sample_d(s)
        return int(s), int(d)

    def loglikelihood(self, z, p=None, x=None, input=None):
        assert z[0] in range(self.K)
        assert z[1] > 0, z
        l_x = self.state_dist.log_prob(z[0])
        if isinstance(self.dur_dist, RecurrentDuration):
            l_d = self.dur_dist.loglikelihood(z[0], z[1], x, input=input)
        else:
            l_d = self.dur_dist.loglikelihood(z[0], z[1])
        return l_x + l_d

    def likelihood(self, z):
        return np.exp(self.loglikelihood(z))

    def update(self, _E):
        raise NotImplementedError

    def report(self):
        report = "initial distribution: K=%s, beta=%s" % (self.K, self.beta)
        log.info(report)
