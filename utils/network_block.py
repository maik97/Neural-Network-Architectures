import torch.nn as nn
import numpy as np
from utils.util_func import maybe_kwargs


class SelfNormLinear(nn.Module):

    def __init__(self, in_features, out_features, linear_kwargs=None):
        super(SelfNormLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **maybe_kwargs(linear_kwargs))

    def reset_parameters(self):
        """
        "For the weight initialization, we propose ω = 0 and τ = 1 for all units in the higher layer"
        """
        nn.init.kaiming_normal_(self.nn_module.weight, mode='fan_in', nonlinearity='linear')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, inputs):
        return self.linear(inputs)


class NetworkBlock(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            nn_module='auto',
            nn_module_kwargs=None,
            activation=None,
            activation_kwargs=None,
            dropout_rate=0.0,
            dropout_type='auto',
            dropout_kwargs=None,
            use_layer_norm=False,
            layer_norm_kwargs=None,
            trace_activations=False,
            self_normalizing=False,
    ):
        super(NetworkBlock, self).__init__()

        ## Main Module
        if nn_module == 'auto':
            nn_module = SelfNormLinear if self_normalizing else nn.Linear
        if not issubclass(nn_module, nn.Module):
            raise TypeError(f'Expected nn_module to be nn.Module, found {nn_module}')
        self.nn_module = nn_module(in_features, out_features, **maybe_kwargs(nn_module_kwargs))

        ## Additional Modules (Activation, Dropout, Layer Norm)
        additional_modules = []

        # Activation
        if activation == 'auto':
            activation = nn.SELU if self_normalizing else nn.ReLU
        if activation is not None:
            additional_modules.append(activation(**maybe_kwargs(activation_kwargs)))

        # Dropout
        if dropout_type == 'auto':
            dropout_type = nn.AlphaDropout if self_normalizing else nn.Dropout
        if dropout_rate != 0.0:
            additional_modules.append(dropout_type(dropout_rate, **maybe_kwargs(dropout_kwargs)))

        # Layer norm
        if use_layer_norm:
            additional_modules.append(nn.LayerNorm(out_features, **maybe_kwargs(layer_norm_kwargs)))

        # Pack Additional Modules
        if len(additional_modules) == 0:
            self.additional_modules = None
        else:
            self.additional_modules = nn.Sequential(*additional_modules)

        ## Parameter
        self.in_features = in_features
        self.out_features = out_features
        self.trace_activations = trace_activations

    def forward(self, inputs):
        x = self.nn_module(inputs)
        if self.trace_activations:
            self.save_activation_distribution(x)
        if self.additional_modules is not None:
            x = self.additional_modules(x)
        return x

    def save_activation_distribution(self, x):
        x = x.detach().numpy()
        if hasattr(self, 'layer_activations'):
            self.layer_activations = np.append(self.layer_activations, x)
        else:
            self.layer_activations = x
