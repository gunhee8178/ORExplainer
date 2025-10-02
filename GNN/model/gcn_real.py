import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv
from .custom_gcn import GCNConvUnweightedDeg

# Graph Convolutional Network (GCN) model for node classification
# 3 layers with ReLU activation
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConvUnweightedDeg(in_channels, hidden_channels)
        self.conv2 = GCNConvUnweightedDeg(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.relu = ReLU()

    # Reset parameters for all layers with seed
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def embed(self, x, edge_index, edge_weight=None):
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = self.relu(x1)
        x1 = F.dropout(x1, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = self.relu(x2)
        x2 = F.dropout(x2, training=self.training)

        return x1, x2

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        _, embedding = self.embed(x, edge_index, edge_weight)
        x = self.lin(embedding)
        return x