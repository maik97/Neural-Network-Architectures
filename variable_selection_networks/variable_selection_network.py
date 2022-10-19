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
        self.num_variables = num_variables

        self.grn_variables = nn.ModuleList()
        for i in range(num_variables):
            self.grn_variables.append(
                GatedResidualNetwork(in_features, out_features, hidden_features, **maybe_kwargs(grn_kwargs))
            )

        self.grn_concat = GatedResidualNetwork(
            in_features=in_features * num_variables,
            out_features=out_features * num_variables,
            hidden_features=hidden_features * num_variables,
            **maybe_kwargs(grn_kwargs)
        )

        self.softmax_block = NetworkBlock(
            in_features=out_features * num_variables,
            out_features=out_features * num_variables,
            **maybe_kwargs(softmax_block_kwargs, defaults=dict(
                activation=None
            ))
        )

        self.softmax = nn.Softmax(-2)

    def forward(self, inputs: torch.Tensor):
        v = self.grn_concat(inputs.view(-1, self.num_variables * self.in_features))
        v = self.softmax(self.softmax_block(v).view(-1, self.num_variables, self.out_features))

        x = []
        inputs = inputs.view(-1, self.num_variables, self.in_features)
        inputs = inputs.transpose(0, 1).split(self.num_variables, 0)
        for var_in, var_grn in zip(inputs, self.grn_variables):
            x.append(var_grn(var_in))

        x = torch.stack(x).view(self.num_variables, -1, self.out_features)
        x = x.transpose(0, 1)

        return v * x


def main():
    import torch
    import numpy as np

    batch_size = 5
    seq_len = 7
    feature_size = 4
    out_features = 3
    hidden_features = 10

    test_tensor = torch.tensor(np.random.random((batch_size, seq_len, feature_size)), dtype=torch.float)

    test_vsn = VariableSelectionNetwork(
        in_features=feature_size,
        out_features=out_features,
        hidden_features=hidden_features,
        num_variables=seq_len
    )

    print(test_tensor.shape)
    print(test_vsn(test_tensor).shape)


if __name__ == '__main__':
    main()