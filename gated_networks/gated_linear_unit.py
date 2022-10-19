import torch
import torch.nn as nn
from utils.util_func import maybe_kwargs
from utils.network_block import NetworkBlock


class GatedLinearUnit(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            gate_kwargs=None,
            block_kwargs=None,
    ):
        super(GatedLinearUnit, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.gate = NetworkBlock(
            in_features=in_features, out_features=out_features,
            **maybe_kwargs(gate_kwargs, defaults=dict(
                activation=nn.Sigmoid
            ))
        )

        self.block = NetworkBlock(
            in_features=in_features, out_features=out_features,
            **maybe_kwargs(block_kwargs, defaults=dict(
                self_normalizing=True,
                activation='auto',
                use_layer_norm=True,
            ))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs) * self.gate(inputs)


def main():
    import numpy as np

    batch_size = 1
    feature_size = 4
    out_features = 3

    test_tensor = torch.tensor(np.random.random((batch_size, feature_size)), dtype=torch.float)

    glu = GatedLinearUnit(
        in_features=feature_size,
        out_features=out_features,
    )

    print(test_tensor.shape)
    print(glu(test_tensor).shape)

if __name__ == '__main__':
    main()