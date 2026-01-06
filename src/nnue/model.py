# src/nnue/model.py
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN = 256

class NnueNet(nn.Module):
    """
    Training-time NNUE model:
      acc_w = b1 + sum(Emb(f_w))
      acc_b = b1 + sum(Emb(f_b))
      x = concat(acc_stm, acc_nstm)
      x = clamp_relu(x)
      y = linear(x)  -> scalar score (centipawns or normalized)
    """
    def __init__(self, num_features: int, hidden: int = HIDDEN, clamp: float = 127.0):
        super().__init__()
        self.hidden = hidden
        self.clamp = clamp

        # Embedding table: [num_features + 1, hidden] // + 1 for padding, allows to send how many pieces there are
        self.emb = nn.Embedding(num_features + 1, hidden)

        # Accumulator bias: [hidden]
        self.b1 = nn.Parameter(torch.zeros(hidden))

        # Output head: [2*hidden] -> [1]
        self.out = nn.Linear(2 * hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # Small init helps stability; you can tune later
        nn.init.uniform_(self.emb.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.b1)
        nn.init.uniform_(self.out.weight, a=-0.001, b=0.001)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats_w: torch.Tensor, feats_b: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        """
        feats_w: LongTensor [B, K]  HalfKP indices for white-king perspective
        feats_b: LongTensor [B, K]  HalfKP indices for black-king perspective
        stm:     LongTensor [B]     0=white to move, 1=black to move

        Returns:
          FloatTensor [B] predicted score
        """
        # [B, K, H] -> sum -> [B, H]
        acc_w = self.b1 + self.emb(feats_w).sum(dim=1)
        acc_b = self.b1 + self.emb(feats_b).sum(dim=1)

        # Reorder so first half is side-to-move
        stm_is_white = (stm == 0).unsqueeze(1)  # [B,1] boolean
        acc_stm = torch.where(stm_is_white, acc_w, acc_b)
        acc_nstm = torch.where(stm_is_white, acc_b, acc_w)

        # [B, 2H]
        x = torch.cat([acc_stm, acc_nstm], dim=1)

        # Clipped ReLU
        x = torch.clamp(x, min=0.0, max=self.clamp)

        # [B,1] -> [B]
        y = self.out(x).squeeze(1)
        return y
