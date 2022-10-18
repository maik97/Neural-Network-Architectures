import torch.nn as nn
from utils.util_func import maybe_kwargs

def nn_block(
        in_features,
        out_features,
        layer_nn=None,
        layer_kwargs=None,
        activation=None,
        activation_kwargs=None,
        dropout_p=0.0,
        dropout_type=None,
        dropout_kwargs=None,
):

    layer_nn = nn.Linear if layer_nn is None else layer_nn
    if not isinstance(layer_nn, nn.Module):
        raise TypeError(f'Expected layer_nn to be nn.Module, found {type(layer_nn)}')

    seq = [layer_nn(in_features, out_features, **maybe_kwargs(layer_kwargs))]

    activation = nn.ReLU if activation == 'auto' else activation
    if activation is not None:
        seq.append(activation(**maybe_kwargs(activation_kwargs)))

    dropout_type = nn.Dropout if dropout_type == 'auto' else dropout_type
    if dropout_type is not None and dropout_p != 0.0:
        seq.append(dropout_type(dropout_p, **maybe_kwargs(dropout_kwargs)))

    return nn.Sequential(*seq)
