import torch.nn as nn
import numpy as np
from utils.init_layer import nn_block

class SelfNormalizingNetworkBlock(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            dropout_p=0.0,
            dropout_type='auto',
            dropout_kwargs=None,
            trace_activations=False,
            layer_kwargs=None,
            layer_nn=None,
            activation='auto',
            activation_kwargs=None,
    ):
        super(SelfNormalizingNetworkBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        activation = nn.SELU() if activation == 'auto' else activation
        self.dense = nn_block(in_features, out_features, layer_nn, layer_kwargs, activation, activation_kwargs)

        dropout_type = nn.AlphaDropout if dropout_type == 'auto' else dropout_type
        dropout_kwargs = {} if dropout_kwargs is None else dropout_kwargs
        if dropout_p == 0.0 or dropout_type is None:
            self.dropout = None
        else:
            self.dropout = dropout_type(dropout_p, **dropout_kwargs)

        self.trace_activations = trace_activations

    def reset_parameters(self):
        """
        "For the weight initialization, we propose ω = 0 and τ = 1 for all units in the higher layer"
        """
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, inputs):
        activation = self.dense(inputs)
        if self.trace_activations:
            self.save_activation_distribution(activation)
        if self.dropout is not None:
            activation = self.dropout(activation)
        return activation


    def save_activation_distribution(self, activation):
        activation = activation.detach().numpy()#.mean()
        if hasattr(self, 'layer_activations'):
            self.layer_activations = np.append(self.layer_activations, activation)
        else:
            self.layer_activations = activation



class SelfNormalizingNetwork(nn.Module):

    def __init__(self, input_dim: int, layer_dims: list, dropout: float=0.0, linear_kwargs=None, trace_activations=False):
        super(SelfNormalizingNetwork, self).__init__()

        self.in_features = input_dim
        self.out_features = layer_dims[-1]

        in_features_list = [input_dim] + layer_dims[:-1]
        out_features_list = layer_dims
        self.module = nn.Sequential(
            *[
                SelfNormalizingNetworkBlock(in_features, out_features, dropout, linear_kwargs, trace_activations)
                for in_features, out_features in zip(in_features_list, out_features_list)
            ]
        )

    def forward(self, inputs):
        return self.module(inputs)


def main():
    import gym
    from stable_baselines3 import PPO
    import seaborn as sns
    from matplotlib import pyplot as plt

    from utils.custom_sb3_ac_policy import CustomActorCriticPolicy

    env = gym.make('CartPole-v0')

    network_kwargs = dict(
        input_dim=env.observation_space.shape[-1],
        layer_dims=[64, 64],
        dropout=0.01,
        trace_activations=False,
    )

    policy_kwargs = dict(
        network=SelfNormalizingNetwork,
        network_kwargs=network_kwargs,
        last_layer_dim_pi=64,
        last_layer_dim_vf=64,
    )

    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=3e-4)

    print(model.policy)

    model.learn(100_000)

    model.policy.mlp_extractor.policy_net.module[0].trace_activations = True
    model.policy.mlp_extractor.policy_net.module[1].trace_activations = True
    model.policy.mlp_extractor.value_net.module[0].trace_activations = True
    model.policy.mlp_extractor.value_net.module[1].trace_activations = True

    for i in range(20):
        obs = env.reset()
        done = False
        episode_r = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            #env.render()
            episode_r += rewards
        print('episode_r',i, episode_r)

    sns.displot(
        {
            'policy_layer_1': model.policy.mlp_extractor.policy_net.module[0].layer_activations,
            'policy_layer_2': model.policy.mlp_extractor.policy_net.module[1].layer_activations,
            'value_layer_1': model.policy.mlp_extractor.value_net.module[0].layer_activations,
            'value_layer_2': model.policy.mlp_extractor.value_net.module[1].layer_activations
        },
        kind='kde'
    )
    plt.show()

if __name__ == '__main__':
    main()
