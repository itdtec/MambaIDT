import torch
import torch.nn as nn
from .encoder import MambaEncoderBlock
from .config import load_config
import torch.nn.functional as F

cfg = load_config()["experiment"]

class MambaITD(nn.Module):
    def __init__(self):
        super(MambaITD, self).__init__()
        C = cfg
        D = C["model_dim"]

        # projections
        self.behavior_embed = nn.Embedding(C["behavior_vocab_size"], D)
        self.fc_interval   = nn.Linear(1, D)
        self.fc_stat       = nn.Linear(C["num_stat_features"], D)

        # === here is the change: use MambaEncoderBlock stacks ===
        self.encoder_b = nn.Sequential(
            *[MambaEncoderBlock(D) for _ in range(C["num_layers"])]
        )
        self.encoder_c = nn.Sequential(
            *[MambaEncoderBlock(D) for _ in range(C["num_layers"])]
        )

        # gating & head remain the same
        self.fc_gate    = nn.Linear(D, D)
        self.layer_norm = nn.LayerNorm(D)
        self.dropout    = nn.Dropout(0.3)
        self.fc1        = nn.Linear(2 * D, C["hidden_dim"])
        self.fc2        = nn.Linear(C["hidden_dim"], 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.zero_()
        self.last_gate = None

    def forward(self, S_b, S_c, X):
        E_b = self.behavior_embed(S_b)
        E_c = self.fc_interval(torch.log(S_c.unsqueeze(-1) + 1e-8))
        E_x = self.fc_stat(X).unsqueeze(1).expand(E_b.size(0), E_b.size(1), -1)
        H_b = self.encoder_b(E_b)
        H_c = self.encoder_c(E_c)
        e_x_session = torch.mean(E_x, dim=1)
        G = torch.sigmoid(self.fc_gate(e_x_session)).unsqueeze(1)
        self.last_gate = G
        F_fusion = self.layer_norm(G * H_b + (1 - G) * H_c)
        F_final = torch.cat([F_fusion, E_x], dim=-1)
        H_hidden = self.dropout(F.relu(self.fc1(F_final)))
        logits = self.fc2(H_hidden).squeeze(-1)
        return logits


    def get_gate_regularization_loss(self):
        return torch.mean(self.last_gate * (1 - self.last_gate))