import torch
import torch.nn as nn

class MambaEncoderBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.norm(x + out)
