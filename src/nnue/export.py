# export_mlp_to_rust.py
from __future__ import annotations

import struct
from pathlib import Path

import torch

from nnue.model import NnueNet, HIDDEN, H1, H2  # ensure H1/H2 exist in model.py

NUM_FEATURES = 64 * 12 * 64

MAGIC = b"NNUE"
VERSION = 2

def choose_scale_for_i16(t: torch.Tensor, target_max: int = 32000) -> int:
    mx = float(t.abs().max().item())
    if mx == 0.0:
        return 1
    s = int(target_max / mx)
    return max(1, s)

def main():
    ckpt = Path("data/processed/nnue_mlp_epoch4.pt")  # update checkpoint name
    outp = Path("data/processed/nnue_mlp.bin")
    outp.parent.mkdir(parents=True, exist_ok=True)

    model = NnueNet(num_features=NUM_FEATURES, hidden=HIDDEN)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # --- tensors ---
    emb = model.emb.weight.detach().contiguous()         # [NUM_FEATURES+1, 256]
    b1  = model.b1.detach().contiguous()                 # [256]

    fc1_w = model.fc1.weight.detach().contiguous()       # [H1, 512]
    fc1_b = model.fc1.bias.detach().contiguous()         # [H1]

    fc2_w = model.fc2.weight.detach().contiguous()       # [H2, H1]
    fc2_b = model.fc2.bias.detach().contiguous()         # [H2]

    out_w = model.out.weight.detach().contiguous().view(-1)  # [H2]
    out_b = model.out.bias.detach().contiguous().view(-1)[0] # scalar

    # --- choose scales ---
    S  = choose_scale_for_i16(emb,   target_max=32000)   # activation scale
    T1 = choose_scale_for_i16(fc1_w, target_max=32000)
    T2 = choose_scale_for_i16(fc2_w, target_max=32000)
    T3 = choose_scale_for_i16(out_w, target_max=32000)

    # --- quantize ---
    emb_i16 = torch.round(emb * S).clamp(-32768, 32767).short()
    b1_i32  = torch.round(b1  * S).int()

    fc1_w_i16 = torch.round(fc1_w * T1).clamp(-32768, 32767).short()
    fc1_b_i32 = torch.round(fc1_b * (S * T1)).int()

    fc2_w_i16 = torch.round(fc2_w * T2).clamp(-32768, 32767).short()
    fc2_b_i32 = torch.round(fc2_b * (S * T2)).int()

    out_w_i16 = torch.round(out_w * T3).clamp(-32768, 32767).short()
    out_b_i32 = int(round(float(out_b.item()) * (S * T3)))

    # --- sanity checks ---
    num_features_plus_pad = emb_i16.shape[0]
    hidden = emb_i16.shape[1]
    assert hidden == HIDDEN
    assert fc1_w_i16.shape == (H1, 2 * hidden)
    assert fc2_w_i16.shape == (H2, H1)
    assert out_w_i16.numel() == H2

    with open(outp, "wb") as f:
        # header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_features_plus_pad))
        f.write(struct.pack("<I", hidden))
        f.write(struct.pack("<I", H1))
        f.write(struct.pack("<I", H2))
        f.write(struct.pack("<i", int(S)))
        f.write(struct.pack("<i", int(T1)))
        f.write(struct.pack("<i", int(T2)))
        f.write(struct.pack("<i", int(T3)))

        # payload
        f.write(emb_i16.numpy().tobytes(order="C"))
        f.write(b1_i32.numpy().tobytes(order="C"))

        f.write(fc1_w_i16.numpy().tobytes(order="C"))
        f.write(fc1_b_i32.numpy().tobytes(order="C"))

        f.write(fc2_w_i16.numpy().tobytes(order="C"))
        f.write(fc2_b_i32.numpy().tobytes(order="C"))

        f.write(out_w_i16.numpy().tobytes(order="C"))
        f.write(struct.pack("<i", int(out_b_i32)))

    print("exported:", outp.resolve())
    print(
        "num_features+pad:", num_features_plus_pad,
        "hidden:", hidden,
        "H1:", H1, "H2:", H2,
        "S:", S, "T1:", T1, "T2:", T2, "T3:", T3
    )

if __name__ == "__main__":
    main()
