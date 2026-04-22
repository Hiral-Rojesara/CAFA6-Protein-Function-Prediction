import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class ProteinFunctionModel(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            ResidualBlock(512),
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
