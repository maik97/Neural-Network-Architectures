import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.util_func import maybe_default_kwarg
from utils.network_block import NetworkBlock

from gated_networks.gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            hidden_features,
            num_variables,
            softmax_dense_kwargs,
            *args, **kwargs,
    ):
        super(VariableSelectionNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.grn_variables = nn.ModuleList()
        for i in range(num_variables):
            self.grn_variables.append(
                GatedResidualNetwork(in_features, out_features, hidden_features, *args, **kwargs)
            )

        self.grn_concat = GatedResidualNetwork(
            in_features=in_features * num_variables,
            out_features=out_features * num_variables,
            hidden_features=hidden_features * num_variables,
            *args, **kwargs
        )

        softmax_dense_kwargs = maybe_default_kwarg(softmax_dense_kwargs, 'activation', nn.Softmax)
        self.softmax_dense = NetworkBlock(
            in_features=in_features * num_variables,
            out_features=out_features * num_variables,
            **softmax_dense_kwargs
        )

    def forward(self, inputs):
        v = torch.cat(inputs)
        v = self.grn_concat(v)
        v = torch.unsqueeze(self.softmax(v), dim=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = torch.stack(x, dim=1)

        return torch.matmul(v.transpose(-1, -2), x).squeeze()
