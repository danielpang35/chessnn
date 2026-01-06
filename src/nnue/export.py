#export to RUST
from __future__ import annotations

import struct
from pathlib import Path

import torch

from nnue.model import NnueNet, HIDDEN

NUM_FEATURES = 64 * 12 * 64

MAGIC = b"NNUE"
VERSION = 1

def choose_scale_for_i16(t: torch.Tensor, target_max: int = 32000) -> int:
    mx = float(t.abs().max().item())
    if mx == 0.0:
        return 1
    s = int(target_max / mx)
    return max(1, s)

def main():
    # pick your checkpoint
    ckpt = Path("data/processed/nnue_epoch4.pt")
    outp = Path("data/processed/nnue.bin")
    outp.parent.mkdir(parents=True, exist_ok=True)

    model = NnueNet(num_features=NUM_FEATURES, hidden=HIDDEN)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    emb = model.emb.weight.detach().contiguous()   # [NUM_FEATURES+1, 256]
    b1  = model.b1.detach().contiguous()           # [256]
    outw = model.out.weight.detach().contiguous().view(-1)  # [512]
    outb = model.out.bias.detach().contiguous().view(-1)[0] # scalar

    # scales chosen to fit i16
    S = choose_scale_for_i16(emb, target_max=32000)   # embedding scale
    T = choose_scale_for_i16(outw, target_max=32000)  # output scale

    emb_i16 = torch.round(emb * S).clamp(-32768, 32767).short()
    b1_i32  = torch.round(b1  * S).int()
    outw_i16 = torch.round(outw * T).clamp(-32768, 32767).short()
    outb_i32 = int(round(float(outb.item()) * (S * T)))

    num_features_plus_pad = emb_i16.shape[0]
    hidden = emb_i16.shape[1]
    assert hidden == HIDDEN
    assert outw_i16.numel() == 2 * hidden

    with open(outp, "wb") as f:
        # header: MAGIC(4), VERSION(u32), num_feat(u32), hidden(u32), S(i32), T(i32)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_features_plus_pad))
        f.write(struct.pack("<I", hidden))
        f.write(struct.pack("<i", int(S)))
        f.write(struct.pack("<i", int(T)))

        # emb: i16 array
        f.write(emb_i16.numpy().tobytes(order="C"))

        # b1: i32 array
        f.write(b1_i32.numpy().tobytes(order="C"))

        # outw: i16 array
        f.write(outw_i16.numpy().tobytes(order="C"))

        # outb: i32 scalar
        f.write(struct.pack("<i", int(outb_i32)))

    print("exported:", outp.resolve())
    print("num_features+pad:", num_features_plus_pad, "hidden:", hidden, "S:", S, "T:", T)

if __name__ == "__main__":
    main()
