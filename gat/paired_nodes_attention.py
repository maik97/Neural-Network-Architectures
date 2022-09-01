import torch
from torch import nn


class ConcatPairedNodeAttention(nn.Module):
    
    def __init__(
            self,
            in_features,
            out_features,
            leaky_relu_negative_slope=0.2,
            use_same_projector=False,
            batch_first=False,
            aggregation_type='sum',
            values_type='new', # 'cat', 'new', 'same'
            update_type='sum', # None, 'cat', 'gru', 'mul', 'sum', 'mean', 'max', 'min'
            use_update_source_projector=False,
            return_only_source_nodes=False,
            edges=None,
            debugging=False,
    ):
        super(ConcatPairedNodeAttention, self).__init__()

        if use_same_projector:
            self.projector = nn.Linear(in_features, out_features)
        else:
            self.source_projector = nn.Linear(in_features, out_features)
            self.target_projector = nn.Linear(in_features, out_features)

        if values_type == 'new':
            self.value_projector = nn.Linear(in_features, out_features)
        elif values_type == 'cat':
            self.value_projector = nn.Linear(in_features * 2, out_features)
        elif values_type == 'gru':
            self.value_gru = nn.GRUCell(in_features, in_features)
            self.value_projector = nn.Linear(in_features, out_features)

        self.attention = nn.Linear(out_features * 2, 1)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)

        self.use_same_projector = use_same_projector
        self.batch_first = batch_first
        self.aggregation_type = aggregation_type
        self.values_type = values_type
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
        self.num_edges = edges.shape[0]
        edges = edges.type(torch.int64)
        edges = edges.transpose(0, 1)
        # Split indices of source and target nodes for each edge
        source_node_id_per_edge = edges[0]
        target_node_id_per_edge = edges[1]
        # Map new indices for sources and targets each
        self.unique_source_node_id, self.source_id_per_edge = self.unique_ids(source_node_id_per_edge)
        self.unique_target_node_id, self.target_id_per_edge = self.unique_ids(target_node_id_per_edge)

        if self.debugging:
            print('edges', edges)
            print('source_node_id_per_edge', source_node_id_per_edge)
            print('target_node_id_per_edge', target_node_id_per_edge)
            print('unique_source_node_id', self.unique_source_node_id)
            print('unique_target_node_id', self.unique_target_node_id)
            print('source_id_per_edge', self.source_id_per_edge)
            print('target_id_per_edge', self.target_id_per_edge)

    def project_node_states(self, nodes):
        if self.use_same_projector:
            nodes = self.projector(nodes)
            source_nodes = nodes[self.unique_source_node_id]
            target_nodes = nodes[self.unique_target_node_id]
        else:
            source_nodes = self.source_projector(nodes[self.unique_source_node_id])
            target_nodes = self.target_projector(nodes[self.unique_target_node_id])
        return source_nodes, target_nodes

    def gather_node_pairs(self, source_nodes, target_nodes):
        source_nodes_expanded = source_nodes[self.source_id_per_edge]
        target_nodes_expanded = target_nodes[self.target_id_per_edge]
        paired_nodes = torch.cat([source_nodes_expanded, target_nodes_expanded], dim=-1)
        return paired_nodes.transpose(0, 1)

    def neighbourhood_softmax(self, attention_scores, batch_size):
        # prepare ids:
        self.batched_ids_sources = self.source_id_per_edge.repeat(batch_size, 1).reshape(batch_size, -1, 1)
        # exp(att):
        attention_scores = torch.exp(torch.clip(attention_scores, -2, 2))
        # placeholder:
        scatter_placeholder = torch.zeros((batch_size, self.num_edges, 1))
        # sum neighbour scores:
        attention_scores_sum_1 = scatter_placeholder.scatter_add(1, self.batched_ids_sources, attention_scores)
        # assign summed neighbour scores to each edge
        attention_scores_sum_2 = attention_scores_sum_1.gather(1, self.batched_ids_sources)
        # calculate softmax for each edge
        attention_scores_norm = attention_scores / attention_scores_sum_2

        if self.debugging:
            print('batched_ids_sources', self.batched_ids_sources.shape, self.batched_ids_sources)
            print('attention_scores', attention_scores.shape, attention_scores)
            print('1) attention_scores_sum', scatter_placeholder.shape, scatter_placeholder)
            print('2) attention_scores_sum', attention_scores_sum_1.shape, attention_scores_sum_1)
            print('3) attention_scores_sum', attention_scores_sum_2.shape, attention_scores_sum_2)
            print('attention_scores_norm', attention_scores_norm.shape, attention_scores_norm)
        return attention_scores_norm

    def apply_attention(self, attention_scores_norm, nodes, target_nodes):
        if self.values_type == 'same':
            values = target_nodes
        elif self.values_type == 'new':
            values = self.value_projector(nodes[self.unique_target_node_id])
        elif self.values_type == 'cat':
            source_values = nodes[self.unique_source_node_id]
            target_values = nodes[self.unique_target_node_id]
            paired_values = torch.cat([source_values, target_values], dim=-1)
            values = self.value_projector(paired_values)
        elif self.values_type == 'gru':
            source_values = nodes[self.unique_source_node_id]
            target_values = nodes[self.unique_target_node_id]
            gru_values = self.value_gru(source_values, target_values)
            values = self.value_projector(gru_values)
        else:
            raise ValueError(f"Invalid values_type: {self.values_type}.")

        values_1 = values[self.target_id_per_edge]
        values_2 = values_1.transpose(0, 1)
        values_3 = attention_scores_norm * values_2

        if self.debugging:
            print('1) values', values_1)
            print('2) values', values_2)
            print('3) values', values_3)

        return values_3

    def aggregate_segment(self, segment):
        if self.aggregation_type == 'sum':
            return segment.sum(0)
        elif self.aggregation_type == 'mean':
            return segment.mean(0)
        elif self.aggregation_type == 'max':
            return segment.max(0)
        else:
            raise ValueError(f"Invalid aggregation_type: {self.aggregation_type}.")

    def aggregate_neighbours(self, values, batch_size):
        # sequence first:
        values = values.transpose(0, 1)

        num_source_nodes = len(self.unique_source_node_id)
        aggregated_values = torch.zeros((num_source_nodes, batch_size, self.out_features))
        compare_src_index = 0
        temp_list = []

        for src_id, targ_id in zip(self.source_id_per_edge, self.target_id_per_edge):
            if self.source_id_per_edge[compare_src_index] == src_id:
                temp_list.append(values[targ_id])

            if len(self.source_id_per_edge) <= compare_src_index + 1:
                segment_done = True
            elif self.source_id_per_edge[compare_src_index + 1] != src_id:
                segment_done = True
            else:
                segment_done = False

            if segment_done:
                segment = torch.stack(temp_list)
                aggregated_values[src_id] = self.aggregate_segment(segment)
                temp_list = []
            compare_src_index += 1

        return aggregated_values

    def update_sources(self, source_nodes, values):
        if self.update_type is None:
            return values
        elif self.update_type == 'cat':
            combined = torch.cat([source_nodes, values], dim=-1)
            return self.concat_update(combined)
        elif self.update_type == 'gru':
            return self.self.gru_update(source_nodes, values)
        elif self.update_type == 'mul':
            return source_nodes * values
        elif self.update_type == 'sum':
            return source_nodes + values
        elif self.update_type == 'mean':
            return (source_nodes + values) / 2
        elif self.update_type == 'max':
            return torch.maximum(source_nodes, values)
        elif self.update_type == 'min':
            return torch.minimum(source_nodes, values)
        else:
            raise ValueError(f"Invalid update_type: {self.aggregation_type}.")

    def forward(self, nodes, edges=None):

        if self.batch_first:
            nodes = nodes.transpose(0, 1)
        num_nodes, batch_size, feature_dim = nodes.shape

        # (0) Prepare indexing stuff
        if edges is not None:
            self.map_node_indices(edges)

        # (1) Node Pairs
        source_nodes, target_nodes = self.project_node_states(nodes)
        paired_nodes = self.gather_node_pairs(source_nodes, target_nodes)

        # (2) Pairwise Attention
        attention_scores = self.activation(self.attention(paired_nodes))

        # (3) Normalize Attention Scores
        attention_scores_norm = self.neighbourhood_softmax(attention_scores, batch_size)

        # (4) Apply Attention
        values = self.apply_attention(attention_scores_norm, nodes, target_nodes)

        # (5) Aggregate Neighbours (Target Nodes)
        aggregated_values = self.aggregate_neighbours(values, batch_size)

        # (6) Update Source Nodes
        if self.use_update_source_projector:
            source_nodes = self.update_source_projector(nodes[self.unique_source_node_id])
        source_nodes = self.update_sources(source_nodes, aggregated_values)

        # (7) Finally return nodes
        if self.return_only_source_nodes:
            out = source_nodes
        else:
            out = torch.zeros((num_nodes, batch_size, self.out_features))[self.unique_target_node_id] = target_nodes
            out[self.unique_source_node_id] = source_nodes

        if self.batch_first:
            out = out.transpose(0, 1)

        return out


def test_node_pairing(batch_size=2, in_features=3,):
    edges = torch.Tensor([[0, 1], [0, 2], [1, 2], [1, 3], [4, 0], [4, 4], [4, 5]])
    nodes = torch.rand(batch_size, 6, in_features)

    batch_size, num_nodes, feature_dim = nodes.shape
    num_edges, _ = edges.shape

    nodes = nodes.transpose(0, 1)
    edges = edges.transpose(0, 1)
    print('nodes', nodes.shape, nodes)
    print('edges', edges.shape, edges)

    ids_sources = edges[0].type(torch.int64)
    ids_targets = edges[1].type(torch.int64)
    print('ids_sources', ids_sources)
    print('ids_targets', ids_targets)

    source_nodes = nodes[ids_sources]
    target_nodes = nodes[ids_targets]

    print(source_nodes)
    print(target_nodes)

    paired_nodes = torch.cat([source_nodes, target_nodes], dim=-1)

    print('paired_nodes', paired_nodes)


def main():
    #test_node_pairing()
    batch_size = 2
    in_features = 5
    out_features = 3
    edges = torch.Tensor([[0, 1], [0, 0], [0, 2], [3, 0], [3, 4], [3, 5], [1, 2], [1, 3]])
    nodes = torch.rand(batch_size, 6, in_features)

    gat = ConcatPairedNodeAttention(in_features, out_features, batch_first=True, debugging=False, edges=edges)
    print(gat(nodes))


if __name__ == '__main__':
    main()
