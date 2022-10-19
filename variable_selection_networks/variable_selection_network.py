import torch
import torch.nn as nn
from utils.util_func import maybe_kwargs
from utils.network_block import NetworkBlock

from gated_networks.gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            hidden_features,
            num_variables,
            grn_kwargs=None,
            softmax_block_kwargs=None,
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
            **maybe_kwargs(grn_kwargs)
        )

        self.softmax_block = NetworkBlock(
            in_features=in_features * num_variables,
            out_features=out_features * num_variables,
            **maybe_kwargs(softmax_block_kwargs, defaults=dict(
                activation=nn.Softmax
            ))
        )

    def forward(self, inputs):
        v = torch.cat(inputs)
        v = self.grn_concat(v)
        v = torch.unsqueeze(self.softmax_block(v), dim=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = torch.stack(x, dim=1)

        return torch.matmul(v.transpose(-1, -2), x).squeeze()
