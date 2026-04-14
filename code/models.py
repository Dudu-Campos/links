import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv



class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # Layer 1: Aggregates neighbors to create hidden representations
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        # Layer 2: Final embedding for classification
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x