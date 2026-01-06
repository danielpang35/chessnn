# export_mlp_to_rust.py
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

from nnue.model import NnueNet, HIDDEN, H1, H2  # ensure H1/H2 exist in model.py

NUM_FEATURES = 64 * 12 * 64

MAGIC = b"NNUE"
# Version with only the MLP head.
VERSION = 2
# Version that appends a fast linear head for qsearch stand-pat/static pruning.
VERSION_WITH_FAST_HEAD = 3

DEFAULT_CKPT = Path("data/processed/nnue_mlp_epoch4.pt")
DEFAULT_OUT = Path("data/processed/nnue_mlp.bin")


def choose_scale_for_i16(t: torch.Tensor, target_max: int = 32000) -> int:
    mx = float(t.abs().max().item())
    if mx == 0.0:
        return 1
    s = int(target_max / mx)
    return max(1, s)


def extract_linear_head(sd: dict[str, torch.Tensor], hidden: int):
    """Extract a fast linear head from a state dict.

    The head must map [acc_stm || acc_nstm] (512 dims) to a scalar, so the weight
    shape must be [1, 2 * hidden].
    """

    candidates = [
        ("fast_out.weight", "fast_out.bias"),  # preferred naming
        ("out.weight", "out.bias"),             # standalone linear head module
    ]

    for w_key, b_key in candidates:
        if w_key not in sd or b_key not in sd:
            continue

        w = sd[w_key]
        b = sd[b_key]

        if w.shape != (1, 2 * hidden):
            continue

        return w.detach().contiguous().view(-1), b.detach().contiguous().view(-1)[0]

    raise ValueError(
        "Fast head checkpoint must contain a linear layer with shape [1, 2 * hidden]"
    )


def main():
    parser = argparse.ArgumentParser(description="Export NNUE MLP (and optional fast head) to Rust binary format")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="MLP checkpoint (full head)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output path for the binary")
    parser.add_argument(
        "--fast-head", type=Path, default=None, help="Optional checkpoint containing a linear fast head (512->1)"
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = NnueNet(num_features=NUM_FEATURES, hidden=HIDDEN)
    sd = torch.load(args.ckpt, map_location="cpu")
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

    fast_out_w = None
    fast_out_b = None
    if args.fast_head is not None:
        fast_sd_raw = torch.load(args.fast_head, map_location="cpu")
        fast_sd = fast_sd_raw["state_dict"] if "state_dict" in fast_sd_raw else fast_sd_raw
        fast_out_w, fast_out_b = extract_linear_head(fast_sd, hidden=HIDDEN)

    # --- choose scales ---
    S  = choose_scale_for_i16(emb,   target_max=32000)   # activation scale
    T1 = choose_scale_for_i16(fc1_w, target_max=32000)
    T2 = choose_scale_for_i16(fc2_w, target_max=32000)
    T3 = choose_scale_for_i16(out_w, target_max=32000)
    T_fast = choose_scale_for_i16(fast_out_w, target_max=32000) if fast_out_w is not None else None

    # --- quantize ---
    emb_i16 = torch.round(emb * S).clamp(-32768, 32767).short()
    b1_i32  = torch.round(b1  * S).int()

    fc1_w_i16 = torch.round(fc1_w * T1).clamp(-32768, 32767).short()
    fc1_b_i32 = torch.round(fc1_b * (S * T1)).int()

    fc2_w_i16 = torch.round(fc2_w * T2).clamp(-32768, 32767).short()
    fc2_b_i32 = torch.round(fc2_b * (S * T2)).int()

    out_w_i16 = torch.round(out_w * T3).clamp(-32768, 32767).short()
    out_b_i32 = int(round(float(out_b.item()) * (S * T3)))

    if fast_out_w is not None:
        fast_out_w_i16 = torch.round(fast_out_w * T_fast).clamp(-32768, 32767).short()
        fast_out_b_i32 = int(round(float(fast_out_b.item()) * (S * T_fast)))

    # --- sanity checks ---
    num_features_plus_pad = emb_i16.shape[0]
    hidden = emb_i16.shape[1]
    assert hidden == HIDDEN
    assert fc1_w_i16.shape == (H1, 2 * hidden)
    assert fc2_w_i16.shape == (H2, H1)
    assert out_w_i16.numel() == H2

    version = VERSION_WITH_FAST_HEAD if fast_out_w is not None else VERSION

    with open(args.out, "wb") as f:
        # header
        f.write(MAGIC)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", num_features_plus_pad))
        f.write(struct.pack("<I", hidden))
        f.write(struct.pack("<I", H1))
        f.write(struct.pack("<I", H2))
        f.write(struct.pack("<i", int(S)))
        f.write(struct.pack("<i", int(T1)))
        f.write(struct.pack("<i", int(T2)))
        f.write(struct.pack("<i", int(T3)))
        if fast_out_w is not None:
            f.write(struct.pack("<i", int(T_fast)))

        # payload
        f.write(emb_i16.numpy().tobytes(order="C"))
        f.write(b1_i32.numpy().tobytes(order="C"))

        f.write(fc1_w_i16.numpy().tobytes(order="C"))
        f.write(fc1_b_i32.numpy().tobytes(order="C"))

        f.write(fc2_w_i16.numpy().tobytes(order="C"))
        f.write(fc2_b_i32.numpy().tobytes(order="C"))

        f.write(out_w_i16.numpy().tobytes(order="C"))
        f.write(struct.pack("<i", int(out_b_i32)))

        if fast_out_w is not None:
            f.write(fast_out_w_i16.numpy().tobytes(order="C"))
            f.write(struct.pack("<i", int(fast_out_b_i32)))

    print("exported:", args.out.resolve())
    print(
        "num_features+pad:", num_features_plus_pad,
        "hidden:", hidden,
        "H1:", H1, "H2:", H2,
        "S:", S, "T1:", T1, "T2:", T2, "T3:", T3,
        "T_fast:", T_fast,
        "version:", version,
    )

if __name__ == "__main__":
    main()
