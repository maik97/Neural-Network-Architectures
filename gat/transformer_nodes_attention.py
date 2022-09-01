import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TransformerNodeAttention(nn.Module):
    
    def __init__(
            self,
            in_features,
            out_features,
            leaky_relu_negative_slope=0.2,
            batch_first=False,
            update_type=None, # None, 'cat', 'gru', 'mul', 'sum', 'mean', 'max', 'min'
            use_update_source_projector=False,
            return_only_source_nodes=False,
            edges=None,
            debugging=False,
    ):
        super(TransformerNodeAttention, self).__init__()

        self.projector = nn.Linear(in_features, out_features)
        self.attention = torch.nn.TransformerEncoder(nn.TransformerEncoderLayer(
                out_features,
                nhead=1,
                dim_feedforward=out_features,
                dropout=0.0,
                batch_first=False
            ),
            num_layers=1,
        )
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        self.batch_first = batch_first
        self.debugging = debugging

        if edges is not None:
            self.map_node_indices(edges)

        self.use_update_source_projector = use_update_source_projector
        if use_update_source_projector:
            self.update_source_projector = nn.Linear(in_features, out_features)

        self.update_type = update_type
        if update_type == 'cat':
            self.concat_update = nn.Linear(out_features * 2, out_features)
        elif update_type == 'gru':
            self.gru_update = nn.GRUCell(out_features, out_features)

        self.return_only_source_nodes = return_only_source_nodes

        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def unique_ids(id_per_edge):
        return torch.unique(id_per_edge, sorted=False, return_inverse=True)

    def map_node_indices(self, edges):
        # Prepare edges
        edges = edges.type(torch.int64)
        edges = edges.transpose(0, 1)
        # Split indices of source and target nodes for each edge
        source_node_id_per_edge = edges[0]
        # Map new indices for sources and targets each
        self.unique_source_node_id, self.source_id_per_edge = self.unique_ids(source_node_id_per_edge)
        self.edges = edges

    def update_sources(self, nodes, values):
        if self.update_type is None:
            return values
        elif self.update_type == 'cat':
            combined = torch.cat([nodes, values], dim=-1)
            return self.concat_update(combined)
        elif self.update_type == 'gru':
            return self.self.gru_update(nodes, values)
        elif self.update_type == 'mul':
            return nodes * values
        elif self.update_type == 'sum':
            return nodes + values
        elif self.update_type == 'mean':
            return (nodes + values) / 2
        elif self.update_type == 'max':
            return torch.maximum(nodes, values)
        elif self.update_type == 'min':
            return torch.minimum(nodes, values)
        else:
            raise ValueError(f"Invalid update_type: {self.update_type}.")

    def forward(self, nodes, edges=None):

        if self.batch_first:
            nodes = nodes.transpose(0, 1)
        num_nodes, batch_size, feature_dim = nodes.shape

        # (0) Prepare indexing stuff
        if edges is not None:
            self.map_node_indices(edges)

        # (1) Node projection
        projected_nodes = self.projector(nodes)

        # (2) Attention mask
        adjacency = np.ones((num_nodes, num_nodes))
        for source_id, target_id in self.edges.transpose(0, 1).numpy():
            adjacency[source_id][target_id] = 0

        for i in range(num_nodes):
            adjacency[i][i] = 0

        attn_mask = torch.Tensor(adjacency).type(torch.bool)#.reshape(1, num_nodes, num_nodes).repeat(batch_size, 1, 1)

        # (3) Multihead Attention
        values = self.attention(projected_nodes, mask=attn_mask)
        values = values.nan_to_num()

        # (4) Update Source Nodes
        if self.use_update_source_projector:
            projected_nodes = self.update_source_projector(projected_nodes)
        updated_nodes = self.update_sources(projected_nodes, values)

        # (7) Finally return nodes
        if self.return_only_source_nodes:
            out = updated_nodes[self.unique_source_node_id]
        else:
            out = updated_nodes

        if self.batch_first:
            out = out.transpose(0, 1)

        return F.relu(out)


def main():
    #test_node_pairing()
    batch_size = 2
    in_features = 5
    out_features = 3
    edges = torch.Tensor([[0, 1], [0, 0], [0, 2], [3, 0], [3, 4], [3, 5], [1, 2], [1, 3]])
    nodes = torch.rand(batch_size, 6, in_features)

    gat = TransformerNodeAttention(in_features, out_features, batch_first=True, debugging=False, edges=edges)
    print(gat(nodes))


if __name__ == '__main__':
    main()
