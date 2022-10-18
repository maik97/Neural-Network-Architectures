import torch.nn as nn
import numpy as np
from utils.init_layer import nn_block

class GatedResidualNetwork(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            hidden_features,
            dropout=0.0,
            elu_layer_nn=None,
            elu_layer_kwargs=None,
            elu_activation_kwargs=None,
            dense_layer_nn=None,
            dense_layer_kwargs=None,
            dense_activation=None,
            dense_activation_kwargs=None,
            dense_dropout_p=0.15,
            dense
    ):
        super(GatedResidualNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = hidden_features

        self.elu_dense = nn_block(
            in_features=in_features,
            out_features=out_features,
            layer_nn=elu_layer_nn,
            layer_kwargs=elu_layer_kwargs,
            activation=nn.ELU,
            activation_kwargs=elu_activation_kwargs,
        )

        self.linear_dense = nn_block(
            in_features=in_features,
            out_features=out_features,
            layer_nn=dense_layer_nn,
            layer_kwargs=dense_layer_kwargs,
            activation=dense_activation,
            activation_kwargs=dense_activation_kwargs,
        )

        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def forward(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x