from torch import nn


class DenseLayer(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            activation=None,
            dropout=None,
            linear_kwargs=None,
            activation_kwargs=None,
            dropout_kwargs=None
    ):
        super(DenseLayer, self).__init__()

        linear_kwargs = {} if linear_kwargs is None else linear_kwargs
        activation_kwargs = {} if activation_kwargs is None else activation_kwargs
        dropout_kwargs = {} if dropout_kwargs is None else dropout_kwargs

        layers = [nn.Linear(in_features, out_features, **linear_kwargs)]

        if activation is not None:
            layers.append(activation(activation, **activation_kwargs))

        if dropout is not None:
            layers.append(dropout(**dropout_kwargs))

        self.module = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.module(inputs)
