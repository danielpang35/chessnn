# src/nnue/dataset.py
from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from nnue.features import extract_features_from_fen

@dataclass
class Sample:
    fw: List[int]
    fb: List[int]
    stm: int
    cp_stm: int       # centipawns from SIDE-TO-MOVE perspective
    y: float          # normalized target (e.g., tanh)

def iter_fen_cp_rows(csv_path: str, max_rows: int | None = None):
    """
    Yield (fen, cp_white) tuples.
    Supports 2-column CSV or alternating-line (fen/cp) files.
    Filters absurd scores (abs(cp) >= 30000).
    """
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        rows = itertools.islice(rdr, max_rows) if max_rows is not None else rdr

        pending_fen = None
        for row in rows:
            if not row:
                continue

            # 2-column: fen, cp
            if len(row) >= 2:
                fen = row[0].strip()
                try:
                    cp = int(float(row[1]))
                except ValueError:
                    # header or invalid line
                    continue
                if abs(cp) >= 30000:
                    continue
                yield fen, cp
                continue

            # 1-column alternating: fen then cp
            cell = row[0].strip()
            if cell.lower() in ("fen", "evaluation"):
                continue

            if pending_fen is None:
                pending_fen = cell
            else:
                try:
                    cp = int(float(cell))
                except ValueError:
                    pending_fen = None
                    continue
                fen = pending_fen
                pending_fen = None
                if abs(cp) >= 30000:
                    continue
                yield fen, cp

class FenCpDataset(Dataset):
    def __init__(self, csv_path: str, max_rows: int | None = None):
        self.samples: List[Sample] = []
        for fen, cp_white in iter_fen_cp_rows(csv_path, max_rows=max_rows):
            pf = extract_features_from_fen(fen)

            # Convert WHITE-relative cp into SIDE-TO-MOVE-relative cp
            cp_stm = cp_white if pf.stm == 0 else -cp_white

            # Normalized target (keep your existing scaling)
            y = math.tanh(cp_stm / 600.0)

            self.samples.append(Sample(
                fw=pf.features_for_white_king(),
                fb=pf.features_for_black_king(),
                stm=pf.stm,
                cp_stm=cp_stm,
                y=y,
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.fw, s.fb, s.stm, s.y, s.cp_stm

def collate_padded(batch):
    fw_list, fb_list, stm_list, y_list, cp_list = zip(*batch)
    max_k = max(len(fw) for fw in fw_list)

    def pad_and_shift(seq):
        # shift by +1 so 0 is reserved for padding in embedding
        out = [x + 1 for x in seq]
        out.extend([0] * (max_k - len(out)))
        return out

    feats_w = torch.tensor([pad_and_shift(fw) for fw in fw_list], dtype=torch.long)
    feats_b = torch.tensor([pad_and_shift(fb) for fb in fb_list], dtype=torch.long)
    stm = torch.tensor(stm_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.float32)
    cp = torch.tensor(cp_list, dtype=torch.int32)  # STM-relative CP for diagnostics
    return feats_w, feats_b, stm, y, cp
