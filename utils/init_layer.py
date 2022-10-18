import torch.nn as nn
from collections import OrderedDict

def nn_block(
        in_features,
        out_features,
        layer_nn=None,
        layer_kwargs=None,
        activation=None,
        activation_kwargs=None,
        dropout_type=None,
        dropout_kwargs=None,
):

    layer_nn = nn.Linear if layer_nn is None else layer_nn
    if not isinstance(layer_nn, nn.Module):
        raise TypeError(f'Expected layer_nn to be nn.Module, found {type(layer_nn)}')

    layer_kwargs = {} if layer_kwargs is None else layer_kwargs
    seq = [layer_nn(in_features, out_features, **layer_kwargs)]

    activation_kwargs = {} if activation_kwargs is None else activation_kwargs
    if activation is not None:
        seq.append(activation(**activation_kwargs))

    dropout_kwargs = {} if dropout_kwargs is None else dropout_kwargs
    if dropout_type is not None:
        seq.append(dropout_type(**dropout_kwargs))

    return nn.Sequential(*seq)
