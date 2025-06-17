import torch
import torch.nn as nn

class MambaEncoderBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(dim, 2 * dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        proj = self.input_proj(x)
        x_proj, gate = proj.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        x_conv = self.conv(x_proj.transpose(1, 2)).transpose(1, 2)
        x_out = self.output_proj(x_conv * gate)
        return self.norm(x_out + residual)
