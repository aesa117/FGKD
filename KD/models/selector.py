import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        if num_layers < 1:
            raise ValueError("number of layers can't be less than 1")
        elif num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Kaiming initialization
        for i, _ in enumerate(self.layers):
            nn.init.kaiming_normal_(self.layers[i].weight)
    
    def forward(self, input):
        hidden = input
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        for l, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if l != len(self.layers) - 1:
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, self.dropout, training=self.training)
        
        return hidden