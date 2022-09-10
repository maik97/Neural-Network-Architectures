from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from utils.layers import DenseLayer


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim,
        network,
        network_kwargs,
        last_layer_dim_pi,
        last_layer_dim_vf,
    ):
        super(CustomNetwork, self).__init__()

        if network is not None:
            self.policy_net = network(**network_kwargs)
            self.value_net = network(**network_kwargs)
        else:
            self.policy_net = nn.Sequential(
                    DenseLayer(feature_dim, 64, nn.ReLU),
                    DenseLayer(64, last_layer_dim_pi, nn.ReLU)
                )
            self.value_net = nn.Sequential(
                    DenseLayer(feature_dim, 64, nn.ReLU),
                    DenseLayer(64, last_layer_dim_vf, nn.ReLU)
                )

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        network = None,
        network_kwargs = None,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        *args,
        **kwargs,
    ):

        self.network = network
        self.network_kwargs = {} if network_kwargs is None else network_kwargs
        self.last_layer_dim_pi = last_layer_dim_pi
        self.last_layer_dim_vf = last_layer_dim_vf

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            feature_dim=self.features_dim,
            network=self.network,
            network_kwargs=self.network_kwargs,
            last_layer_dim_pi=self.last_layer_dim_pi,
            last_layer_dim_vf=self.last_layer_dim_vf,
        )


def main():
    model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
    model.learn(500_000)


if __name__ == '__main__':
    main()
