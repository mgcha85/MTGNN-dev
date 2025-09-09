import torch.nn.functional as F
from layer import *

class net_lstm(nn.Module):
    def __init__(self, lstm_layers, hidden_size, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(net_lstm, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.num_layers=lstm_layers
        self.hidden_size=hidden_size

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length

        for j in range(1,layers+1):
            if self.gcn_true:
                self.gconv1.append(mixprop(seq_length, 32, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(seq_length, 32, gcn_depth, dropout, propalpha))

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, dropout=0.1, batch_first=True)

        self.linear = nn.Linear(64, out_dim)

        self.idx = torch.arange(self.num_nodes).to(device)
        
        self.device=device

    def forward(self, x, idx=None):
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx) # this line computes the adjacency matrix adaptively by calling the function forward in the gc
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        # LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x.view(len(x), x.size(0), -1), (h0, c0))
        out = out[-1, :, :]

        # Graph convolutional layers
        for j in range(self.layers):
            if self.gcn_true:
                x = self.linear[j](out).view(-1, self.node_dim, self.seq_length)
                x = F.relu(self.gconv1[j](x, adp))
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gconv2[j](x, adp)
                x = self.norm[j](x)

        return x.transpose(1,2).contiguous().view(-1, self.num_nodes * self.node_dim)
