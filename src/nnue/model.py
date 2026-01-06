# src/nnue/model.py
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN = 256
H1 = 32
H2 = 32

class NnueNet(nn.Module):
    """
    Training-time NNUE model with an MLP head.

      acc_w = b1 + sum(Emb(f_w))
      acc_b = b1 + sum(Emb(f_b))

      # canonicalize to side-to-move perspective
      x = concat(acc_stm, acc_nstm)              # [B, 512]
      x = clamp_relu(x)

      # MLP head
      h1 = clamp_relu(fc1(x))                    # [B, H1]
      h2 = clamp_relu(fc2(h1))                   # [B, H2]
      y  = out(h2)                               # [B, 1] -> [B]
    """
    def __init__(self, num_features: int, hidden: int = HIDDEN, clamp: float = 127.0):
        super().__init__()
        assert hidden == HIDDEN, "This model assumes HIDDEN=256 for now."

        self.hidden = hidden
        self.clamp = clamp

        # Embedding table: [num_features + 1, hidden] // +1 for padding index 0
        self.emb = nn.Embedding(num_features + 1, hidden)

        # Accumulator bias: [hidden]
        self.b1 = nn.Parameter(torch.zeros(hidden))

        # MLP head
        self.fc1 = nn.Linear(2 * hidden, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.out = nn.Linear(H2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # Embeddings / accumulator bias
        nn.init.uniform_(self.emb.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.b1)

        # Hidden layers: modest init
        nn.init.uniform_(self.fc1.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.fc2.bias)

        # Output: smaller init for stability
        nn.init.uniform_(self.out.weight, a=-0.001, b=0.001)
        nn.init.zeros_(self.out.bias)

    def clamp_relu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=self.clamp)

    def forward(self, feats_w: torch.Tensor, feats_b: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        # [B, K, H] -> sum -> [B, H]
        acc_w = self.b1 + self.emb(feats_w).sum(dim=1)
        acc_b = self.b1 + self.emb(feats_b).sum(dim=1)

        # Side-to-move canonicalization
        stm_is_white = (stm == 0).unsqueeze(1)  # [B,1]
        acc_stm = torch.where(stm_is_white, acc_w, acc_b)
        acc_nstm = torch.where(stm_is_white, acc_b, acc_w)

        # [B, 512]
        x = torch.cat([acc_stm, acc_nstm], dim=1)
        x = self.clamp_relu(x)

        # MLP head
        h = self.clamp_relu(self.fc1(x))
        h = self.clamp_relu(self.fc2(h))

        # [B]
        y = self.out(h).squeeze(1)
        return y
