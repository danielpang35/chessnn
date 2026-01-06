# src/nnue/dataset.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple
import math
import torch
from torch.utils.data import Dataset

from nnue.features import extract_features_from_fen

NUM_FEATURES = 64 * 12 * 64

@dataclass
class Sample:
    fw: List[int]
    fb: List[int]
    stm: int
    y: float

class FenCpDataset(Dataset):
    def __init__(self, csv_path: str, max_rows: int | None = None):
        self.samples: List[Sample] = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Skip header if present
            first = next(reader, None)
            if first is not None:
                # If it looks like a header (second column not numeric), ignore it.
                try:
                    float(first[1])
                    # It was data, not header
                    row = first
                    # fall through by processing row below
                    fen = row[0].strip()
                    y_cp = float(row[1])

                    pf = extract_features_from_fen(fen)
                    self.samples.append(Sample(
                        fw=pf.features_for_white_king(),
                        fb=pf.features_for_black_king(),
                        stm=pf.stm,
                        y=math.tanh(y_cp / 600.0),
                    ))
                except Exception:
                    pass

            for row in reader:
                if not row or len(row) < 2:
                    continue

                fen = row[0].strip()

                try:
                    y_cp = float(row[1])
                except ValueError:
                    # skip any malformed rows / headers
                    continue
                # Drop mate-clamped values for now (you set these to +/-30000)
                if abs(y_cp) >= 30000:
                    continue

                pf = extract_features_from_fen(fen)
                self.samples.append(Sample(
                    fw=pf.features_for_white_king(),
                    fb=pf.features_for_black_king(),
                    stm=pf.stm,
                    y=math.tanh(y_cp / 600.0),
                ))

                if max_rows is not None and len(self.samples) >= max_rows:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # Return variable-length lists; collate_fn will pad/pack if needed.
        return s.fw, s.fb, s.stm, s.y

def collate_padded(batch):
    fw_list, fb_list, stm_list, y_list = zip(*batch)

    max_k = max(len(fw) for fw in fw_list)

    # Pad with 0, but shift real indices by +1 (0 reserved for padding)
    def pad_and_shift(seq):
        out = [x + 1 for x in seq]
        out.extend([0] * (max_k - len(out)))
        return out

    feats_w = torch.tensor([pad_and_shift(fw) for fw in fw_list], dtype=torch.long)
    feats_b = torch.tensor([pad_and_shift(fb) for fb in fb_list], dtype=torch.long)

    stm = torch.tensor(stm_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.float32)
    return feats_w, feats_b, stm, y