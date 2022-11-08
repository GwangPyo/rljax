import gym
import numpy as np


class TestEnv(gym.Env):
    """
    Single-step very simple environment to verify that
    an agent is able to find
    (i) maximizing CVaR 50% action, = 1
    (ii) maximizing  worst case action, = 0
    (iii) maximizing expectation action, = -1
    """
    def __init__(self):
        # dummy
        self.observation_space = gym.spaces.Box(0, 1, shape=(1, ))
        self.action_space = gym.spaces.Box(-1, 1, shape=(1, ))
        self._max_episode_steps = 2

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        if seed:
            return np.random.seed(seed=seed)

    def reset(self):
        return np.zeros(shape=(1, ))

    def step(self, action):
        t = np.random.uniform(0, 1, size=(1, ))[0]
        x = action[0]
        lt_zero = 1. - np.heaviside(x, 0)  # jnp.asarray(x <= 0, dtype=jnp.float32)

        reward = lt_zero * (np.abs(x) * self.g(t) + (1. - np.abs(x)) * self.h(t)) + (1. - lt_zero) * (
                    (1. - np.abs(x)) * self.h(t) + np.abs(x) * self.f(t))
        return np.ones(shape=(1, )), reward, True, {}

    @staticmethod
    def f(t):
        """
        Worst case = -1 (Best among other Threes)
        CVaR 50%   = -1
        Mean       = -1
        """
        return -1.

    @staticmethod
    def g(t):
        """
        Worst case = -2
        CVaR 50%   =  0 (Best among other Threes)
        Mean       =  2
        """
        return 8 * t - 2.

    @staticmethod
    def h(t):
        """
        Worst case = -2.5
        CVaR  50%  = -0.5
        Mean       =  2.5 (Best Among Threes)
        """
        return 6 * ((t + 0.5 ) ** 2) - 4


