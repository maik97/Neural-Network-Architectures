import gym
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TestFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, network, features_dim: int):
        super(TestFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.network = network

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return F.relu(self.network(observations))
