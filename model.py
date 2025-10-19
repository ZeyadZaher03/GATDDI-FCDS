import torch, torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden=256, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden, heads=heads, dropout=dropout))
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
        self.dropout = dropout
        self.scorer = nn.Sequential(
            nn.Linear(hidden * heads * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        h = torch.cat([z[src], z[dst]], dim=-1)
        return self.scorer(h).view(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
