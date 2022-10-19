# Neural Network Architectures

This project hosts a collection of custom neural network architectures I implemented with PyTorch.
I am especially interested in Graph Neural Networks and Transformer architectures.
See below for a list of the implementations so far.

## Graph Neural Networks
- [x] Graph Attention Networks (GAT)
  [[1]](https://arxiv.org/abs/1710.10903)
  - [x] GAT with Paired Nodes Attention
    [[Code]](https://github.com/maik97/Neural-Network-Architectures/blob/main/gat/paired_nodes_attention.py)
  - [x] GAT with Masked Self-Attention
    [[Code]](https://github.com/maik97/Neural-Network-Architectures/blob/main/gat/transformer_nodes_attention.py)
- [ ] Graph Convolutional Networks (GCN)
  [[2]](https://arxiv.org/abs/1609.02907)
  [[3]](https://arxiv.org/abs/1606.09375)

## Gated Neural Networks
- [x] Gated Linear Unit (GLU)
  [[4]](https://arxiv.org/abs/1612.08083)
  [[Code]](https://github.com/maik97/Neural-Network-Architectures/blob/main/gated_networks/gated_linear_unit.py)
- [x] Gated Residual Network (GRN)
  [[5]](https://arxiv.org/abs/1912.09363)
  [[Code]](https://github.com/maik97/Neural-Network-Architectures/blob/main/gated_networks/gated_residual_network.py)

## Variable Selection Neural Networks
- [x] Variable Selection Network (VSN)
  [[5]](https://arxiv.org/abs/1912.09363)
  [[Code]](https://github.com/maik97/Neural-Network-Architectures/blob/main/variable_selection_networks/variable_selection_network.py)

## Other Networks
- [ ] Modern Hopfield Network
  [[6]](https://arxiv.org/abs/2008.02217)
- [X] Self-Normalizing Neural Network
  [[7]](https://arxiv.org/abs/1706.02515)
  [[Code]](https://github.com/maik97/Neural-Network-Architectures/tree/main/self_normalizing_neural_networks)

## References


1. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. doi:10.48550/ARXIV.1710.10903
[[arxiv]](https://arxiv.org/abs/1710.10903)


2. Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks. doi:10.48550/ARXIV.1609.02907
[[arxiv]](https://arxiv.org/abs/1609.02907)


3. Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. doi:10.48550/ARXIV.1606.09375
[[arxiv]](https://arxiv.org/abs/1606.09375)


4. Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2016). Language Modeling with Gated Convolutional Networks. doi:10.48550/ARXIV.1612.08083
[[arxiv]](https://arxiv.org/abs/1612.08083)


5. Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2019). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. doi:10.48550/ARXIV.1912.09363
[[arxiv]](https://arxiv.org/abs/1912.09363)


6. Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., … Hochreiter, S. (2020). Hopfield Networks is All You Need. doi:10.48550/ARXIV.2008.02217
[[arxiv]](https://arxiv.org/abs/2008.02217)


7. Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. doi:10.48550/ARXIV.1706.02515
[[arxiv]](https://arxiv.org/abs/1706.02515)

