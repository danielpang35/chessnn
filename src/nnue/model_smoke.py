# src/nnue/model_smoke.py
import torch
from nnue.features import extract_features_from_fen
from nnue.model import NnueNet

NUM_FEATURES = 64 * 12 * 64  # HalfKP per perspective

def main():
    pf = extract_features_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    fw = pf.features_for_white_king()
    fb = pf.features_for_black_king()

    # Batch size 1, K=32
    feats_w = torch.tensor([fw], dtype=torch.long)
    feats_b = torch.tensor([fb], dtype=torch.long)
    stm = torch.tensor([pf.stm], dtype=torch.long)

    model = NnueNet(num_features=NUM_FEATURES)
    y = model(feats_w, feats_b, stm)
    print("output:", y, "shape:", y.shape)

if __name__ == "__main__":
    main()
