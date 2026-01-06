# src/nnue/train.py
from __future__ import annotations

import random
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from nnue.dataset import FenCpDataset, collate_padded
from nnue.model import NnueNet

NUM_FEATURES = 64 * 12 * 64

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FenCpDataset("data/raw/FilteredEvals.csv")

    idx = list(range(len(ds)))
    random.Random(1234).shuffle(idx)
    split = int(0.9 * len(ds))

    train_ds = Subset(ds, idx[:split])
    val_ds = Subset(ds, idx[split:])

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, collate_fn=collate_padded)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_padded)

    model = NnueNet(num_features=NUM_FEATURES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # For now, keep simple MSE on normalized y. (We can add weighting later.)
    loss_fn = torch.nn.MSELoss()

    def run_eval():
        model.eval()
        total = 0.0
        cnt = 0
        with torch.no_grad():
            for feats_w, feats_b, stm, y, cp in val_dl:
                feats_w = feats_w.to(device)
                feats_b = feats_b.to(device)
                stm = stm.to(device)
                y = y.to(device)

                pred = model(feats_w, feats_b, stm)
                loss = loss_fn(pred, y)

                total += loss.item() * y.size(0)
                cnt += y.size(0)

        model.train()
        return total / cnt

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    for epoch in range(5):
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        running = 0.0
        seen = 0

        for feats_w, feats_b, stm, y, cp in pbar:
            feats_w = feats_w.to(device)
            feats_b = feats_b.to(device)
            stm = stm.to(device)
            y = y.to(device)

            pred = model(feats_w, feats_b, stm)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * y.size(0)
            seen += y.size(0)
            pbar.set_postfix(train_mse=running / seen)

        val_mse = run_eval()
        print(f"epoch {epoch} val_mse={val_mse:.6f} val_rmse={math.sqrt(val_mse):.6f}")

        torch.save(model.state_dict(), f"data/processed/nnue_mlp_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
