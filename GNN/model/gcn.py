import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class NodeGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, hidden_dim, num_classes, device=None):
        super(NodeGCN, self).__init__()
        self.embedding_size = hidden_dim * 3
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu2 = ReLU()


        self.dropout = torch.nn.Dropout(0.5)
        self.lin = Linear(hidden_dim, num_classes)

        self.device=device
    def forward(self, x, edge_index, edge_weight=None):
        input_lin = self.embedding(x, edge_index, edge_weight)
        final = self.lin(input_lin)
        return input_lin, final

    def embedding(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=self.device)

        out1 = self.conv1(x, edge_index, edge_weight)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        out1 = self.dropout(out1)

        out2 = self.conv2(out1, edge_index, edge_weight)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        out2 = self.dropout(out2)

        return out2

class GraphGCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GraphGCN, self).__init__()
        self.embedding_size = hidden_dim
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size * 2, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index, edge_weight)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out, input_lin

    def embedding(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weight)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weight)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weight)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin
