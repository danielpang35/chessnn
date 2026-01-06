# src/nnue/dataset.py
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from typing import List
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


def iter_fen_cp_rows(csv_path: str, max_rows: int | None = None):
    """Yield (fen, centipawn) tuples from either 2-column CSVs or alternating-line files.

    Some public datasets ship as a single-column CSV where FENs and centipawn values
    alternate line by line (with optional ``FEN``/``Evaluation`` headers). Others use a
    traditional 2-column CSV. This helper normalizes both formats.
    """

    def _yield_column_rows(rows):
        count = 0
        for row in rows:
            if max_rows is not None and count >= max_rows:
                break

            if not row or len(row) < 2:
                continue

            fen = row[0].strip()
            try:
                y_cp = float(row[1])
            except (TypeError, ValueError):
                continue

            if abs(y_cp) >= 30000:
                continue

            count += 1
            yield fen, y_cp

    def _yield_alternating_rows(rows):
        count = 0
        pending_fen: str | None = None

        for row in rows:
            if max_rows is not None and count >= max_rows:
                break

            if not row:
                continue

            cell = row[0].strip()
            if not cell:
                continue

            lower = cell.lower()
            if lower in {"fen", "evaluation"}:
                continue

            if pending_fen is None:
                pending_fen = cell
                continue

            try:
                y_cp = float(cell)
            except ValueError:
                # Treat the current line as the next FEN candidate.
                pending_fen = cell
                continue

            if abs(y_cp) >= 30000:
                pending_fen = None
                continue

            count += 1
            yield pending_fen, y_cp
            pending_fen = None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        buffered = []
        for _ in range(5):
            try:
                buffered.append(next(reader))
            except StopIteration:
                break

        row_iter = itertools.chain(buffered, reader)
        has_columns = any(len(r) >= 2 for r in buffered if r)

        yield from _yield_column_rows(row_iter) if has_columns else _yield_alternating_rows(row_iter)

class FenCpDataset(Dataset):
    def __init__(self, csv_path: str, max_rows: int | None = None):
        self.samples: List[Sample] = []
        for fen, y_cp in iter_fen_cp_rows(csv_path, max_rows=max_rows):
            pf = extract_features_from_fen(fen)
            self.samples.append(Sample(
                fw=pf.features_for_white_king(),
                fb=pf.features_for_black_king(),
                stm=pf.stm,
                y=math.tanh(y_cp / 600.0),
            ))

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
