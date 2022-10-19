import torch
import torch.nn as nn
from utils.util_func import maybe_kwargs
from utils.network_block import NetworkBlock


class FloatVariableEncoderUnit(nn.Module):

    def __init__(self, in_features, out_features, *args, **kwargs):
        super(FloatVariableEncoderUnit, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block = NetworkBlock(1, out_features, *args, **kwargs)

    def forward(self, inputs: torch.Tensor):
        if inputs.shape[-1] != 1:
            inputs = inputs.unsqueeze(-1)
        return self.block(inputs)


class FloatVariableEncoderMultiHead(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            num_heads=4,
            fveu_kwargs=None,
            attention_kwargs=None,
            use_input_projector=True,
            projector_kwargs=None,
    ):
        super(FloatVariableEncoderMultiHead, self).__init__()

        if num_heads == 1:
            raise Exception("Use FloatVariableEncoderUnit if you only want to use 1 head.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.multi_head_block = FloatVariableEncoderUnit(
            in_features=in_features, out_features=out_features * num_heads,
            **maybe_kwargs(fveu_kwargs, defaults=dict(
                self_normalizing=True,
                activation='auto',
                use_layer_norm=True,
            ))
        )

        self.head_attention = NetworkBlock(
            in_features=in_features, out_features=in_features * num_heads,
            **maybe_kwargs(attention_kwargs, defaults=dict(
                self_normalizing=True,
                activation='auto',
            ))
        )

        self.softmax = nn.Softmax(dim=-2)

        if use_input_projector:
            self.projector = NetworkBlock(
                in_features=in_features, out_features=in_features,
                **maybe_kwargs(projector_kwargs, defaults=dict(
                    self_normalizing=True,
                    activation='auto',
                    use_layer_norm=True,
                ))
            )
        else:
            self.projector = None

    def forward(self, inputs: torch.Tensor):

        if self.projector is not None:
            inputs = self.projector(inputs)

        in_shape = inputs.shape
        x = self.multi_head_block(inputs).view(in_shape+(self.num_heads, self.out_features))
        att = self.head_attention(inputs).view(in_shape+(self.num_heads, 1))
        scores = self.softmax(att)

        return torch.sum(x * scores, dim=-2)


def main():
    import numpy as np

    batch_size = 5
    feature_size = 4
    out_features = 3
    num_heads = 10

    test_tensor = torch.tensor(np.random.random((batch_size, feature_size)), dtype=torch.float)

    test_fveu_mh = FloatVariableEncoderMultiHead(
        in_features=feature_size,
        out_features=out_features,
        num_heads=num_heads
    )

    print(test_tensor.shape)
    print(test_fveu_mh(test_tensor).shape)


if __name__ == '__main__':
    main()
