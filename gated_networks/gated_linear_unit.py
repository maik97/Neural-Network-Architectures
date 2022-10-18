import torch.nn as nn
from utils.init_layer import nn_block
from utils.util_func import maybe_kwargs, maybe_default_kwarg


class GatedLinearUnit(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            gate_kwargs=None,
            dense_kwargs=None,
    ):
        super(GatedLinearUnit, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        gate_kwargs = maybe_default_kwarg(gate_kwargs, 'activation', nn.Sigmoid)
        self.gate = nn_block(in_features, out_features, **gate_kwargs)
        self.dense = nn_block(in_features, out_features, **maybe_kwargs(dense_kwargs))

    def forward(self, inputs):
        return self.dense(inputs) * self.gate(inputs)
