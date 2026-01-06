from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import chess
import chess.pgn

# ---- CONFIG (defaults; override via CLI) ----
# Point this to your Stockfish .exe
STOCKFISH_PATH = r"../stockfish-windows-x86-64-avx2.exe"

# Output CSV (project-root relative). Do NOT use ../../ paths.
OUT_CSV = Path("data/raw/positions.csv")

# Evaluation settings
DEPTH = 15
MATE_CP = 30000

# PGN settings
# Set this to your PGN file (or a folder; see below)
PGN_PATH = Path("data/raw/games.pgn")  # change to your actual PGN path

# Sampling: write at most this many labeled positions
MAX_POSITIONS = 50_000

# Only start sampling after N plies (avoid openings) and stop after N plies (avoid long games)
MIN_PLY = 8
MAX_PLY = 140

# Sample every N plies (e.g. 2 = every 2 plies)
SAMPLE_EVERY = 2

# How often to force flush to disk
FSYNC_EVERY = 100


class StockfishUCI:
    """
    Persistent Stockfish process (much faster than spawning per FEN).
    """

    def __init__(self, exe_path: str, mate_cp: int = MATE_CP):
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin and self.proc.stdout
        self.mate_cp = mate_cp
        self._uci_init()

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        return line

    def _uci_init(self) -> None:
        self._send("uci")
        # Wait for uciok
        while True:
            line = self._readline()
            if not line:
                raise RuntimeError("Stockfish terminated during uci init")
            if "uciok" in line:
                break

        self._send("isready")
        while True:
            line = self._readline()
            if not line:
                raise RuntimeError("Stockfish terminated during isready")
            if "readyok" in line:
                break

    def eval_cp(self, fen: str, depth: int = DEPTH) -> int:
        """
        Returns evaluation in centipawns from side-to-move perspective.
        Mates mapped to +/- MATE_CP.
        """
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        last_cp = 0
        last_mate: Optional[int] = None

        while True:
            line = self._readline()
            if not line:
                raise RuntimeError("Stockfish terminated during search")

            if line.startswith("info"):
                parts = line.split()
                # Look for "... score cp X ..." or "... score mate N ..."
                if "score" in parts:
                    i = parts.index("score")
                    if i + 2 < len(parts):
                        kind = parts[i + 1]
                        val = parts[i + 2]
                        if kind == "cp":
                            try:
                                last_cp = int(val)
                            except ValueError:
                                pass
                        elif kind == "mate":
                            try:
                                last_mate = int(val)
                            except ValueError:
                                pass

            if line.startswith("bestmove"):
                break

        if last_mate is not None:
            return self.mate_cp if last_mate > 0 else -self.mate_cp
        return last_cp

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def iter_pgn_games(pgn_path: Path) -> Iterable[chess.pgn.Game]:
    """
    Yields games from a .pgn file, or all .pgn files in a directory.
    """
    paths: Sequence[Path]
    if pgn_path.is_dir():
        paths = sorted(pgn_path.glob("*.pgn"))
    else:
        paths = [pgn_path]

    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game


def iter_sampled_positions_from_game(
    game: chess.pgn.Game,
    min_ply: int = MIN_PLY,
    max_ply: int = MAX_PLY,
    sample_every: int = SAMPLE_EVERY,
) -> Iterable[str]:
    """
    Generates FENs from a single game by replaying moves and sampling positions.
    """
    board = game.board()
    ply = 0

    for move in game.mainline_moves():
        board.push(move)
        ply += 1

        if ply < min_ply:
            continue
        if ply > max_ply:
            break
        if (ply - min_ply) % sample_every != 0:
            continue

        # Exclude terminal positions (optional, but usually cleaner)
        if board.is_game_over():
            break

        yield board.fen()


def ensure_out_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def count_existing_rows(csv_path: Path) -> int:
    """
    Counts existing rows (excluding header) for append/resume workflows.
    """
    if not csv_path.exists():
        return 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if present
        return sum(1 for _ in reader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stockfish-labeled NNUE positions")
    parser.add_argument("--stockfish", type=str, default=STOCKFISH_PATH, help="Path to Stockfish executable")
    parser.add_argument("--pgn", type=Path, default=PGN_PATH, help="PGN file or directory of PGNs")
    parser.add_argument("--out", type=Path, default=OUT_CSV, help="Output CSV path")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Search depth")
    parser.add_argument("--mate-cp", type=int, default=MATE_CP, help="Clamp value for mate scores")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS, help="Positions to write (per run)")
    parser.add_argument("--min-ply", type=int, default=MIN_PLY, help="Minimum ply before sampling")
    parser.add_argument("--max-ply", type=int, default=MAX_PLY, help="Maximum ply to consider")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY, help="Sample every N plies")
    parser.add_argument("--fsync-every", type=int, default=FSYNC_EVERY, help="Flush and fsync frequency")
    parser.add_argument("--report-every", type=int, default=50, help="Progress print frequency")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV (resume)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output location
    ensure_out_dir(args.out)
    out_abs = args.out.resolve()
    print("Writing dataset to:", out_abs)

    # Validate Stockfish exists
    if not os.path.exists(args.stockfish):
        raise FileNotFoundError(f"Stockfish not found: {args.stockfish}")

    # Validate PGN exists
    if not args.pgn.exists():
        raise FileNotFoundError(f"PGN not found: {args.pgn.resolve()}")

    existing_rows = count_existing_rows(args.out) if args.append else 0
    mode = "a" if args.append else "w"
    write_header = not args.append or existing_rows == 0
    target_total = existing_rows + args.max_positions

    if args.append and existing_rows:
        print(f"Append mode: found {existing_rows} existing rows; target total = {target_total}")
    elif args.append:
        print("Append mode requested but no existing file; starting fresh.")

    engine = StockfishUCI(args.stockfish, mate_cp=args.mate_cp)

    n_written = existing_rows
    n_seen = 0

    try:
        with open(args.out, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["fen", "cp"])

            for game in iter_pgn_games(args.pgn):
                for fen in iter_sampled_positions_from_game(
                    game,
                    min_ply=args.min_ply,
                    max_ply=args.max_ply,
                    sample_every=args.sample_every,
                ):
                    n_seen += 1
                    cp = engine.eval_cp(fen, depth=args.depth)
                    writer.writerow([fen, cp])
                    n_written += 1

                    # Print progress occasionally
                    if n_written % args.report_every == 0:
                        print(f"written={n_written} last_cp={cp}")

                    # Force flush so you always see file growth
                    if args.fsync_every > 0 and n_written % args.fsync_every == 0:
                        f.flush()
                        os.fsync(f.fileno())

                    if n_written >= target_total:
                        print("Reached target count.")
                        f.flush()
                        os.fsync(f.fileno())
                        return

            # Final flush
            f.flush()
            os.fsync(f.fileno())
    finally:
        engine.close()

    print(f"Done. seen={n_seen} written={n_written}")


if __name__ == "__main__":
    main()
