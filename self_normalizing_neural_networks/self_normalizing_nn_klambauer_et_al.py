import torch.nn as nn


class SelfNormalizingNetworkBlock(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.0, linear_kwargs=None):
        super(SelfNormalizingNetworkBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        linear_kwargs = {} if linear_kwargs is None else linear_kwargs
        self.linear = nn.Linear(in_features, out_features, **linear_kwargs)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(p=dropout)

    def reset_parameters(self):
        """
        "For the weight initialization, we propose ω = 0 and τ = 1 for all units in the higher layer"
        """
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, inputs):
        return self.dropout(self.selu(self.linear(inputs)))

    def test_activation(self, inputs):
        return self.selu(self.linear(inputs))


class SelfNormalizingNetwork(nn.Module):

    def __init__(self, input_dim: int, layer_dims: list, dropout: float=0.0, linear_kwargs=None):
        super(SelfNormalizingNetwork, self).__init__()

        self.in_features = input_dim
        self.out_features = layer_dims[-1]

        in_features_list = [input_dim] + layer_dims[:-1]
        out_features_list = layer_dims
        self.module = nn.Sequential(
            *[
                SelfNormalizingNetworkBlock(in_features, out_features, dropout, linear_kwargs)
                for in_features, out_features in zip(in_features_list, out_features_list)
            ]
        )

    def forward(self, inputs):
        return self.module(inputs)

    def test_network_activations(self, inputs):
        activations = []
        for snn_block in self.module:
            inputs = snn_block.test_activation(inputs)
            activations.append(inputs)
        return activations


def main():
    import gym
    from utils.testing_with_sb3 import test_with_ppo

    env = gym.make('CartPole-v0')
    network_kwargs = dict(
        input_dim=env.observation_space.shape[-1],
        layer_dims=[64, 64]
    )
    test_with_ppo(
        env=env,
        network=SelfNormalizingNetwork,
        network_kwargs=network_kwargs,
        last_layer_dim_pi=64,
        last_layer_dim_vf=64,
    )

if __name__ == '__main__':
    main()
