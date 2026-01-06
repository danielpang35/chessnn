from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import chess
import chess.pgn

# ---- CONFIG ----
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

    def __init__(self, exe_path: str):
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin and self.proc.stdout
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
            return MATE_CP if last_mate > 0 else -MATE_CP
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
    Yields games from a .pgn file.
    """
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
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


def main():
    # Resolve output location
    ensure_out_dir(OUT_CSV)
    out_abs = OUT_CSV.resolve()
    print("Writing dataset to:", out_abs)

    # Validate Stockfish exists
    if not os.path.exists(STOCKFISH_PATH):
        raise FileNotFoundError(f"Stockfish not found: {STOCKFISH_PATH}")

    # Validate PGN exists
    if not PGN_PATH.exists():
        raise FileNotFoundError(f"PGN not found: {PGN_PATH.resolve()}")

    engine = StockfishUCI(STOCKFISH_PATH)

    n_written = 0
    n_seen = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "cp"])

        for game in iter_pgn_games(PGN_PATH):
            for fen in iter_sampled_positions_from_game(game):
                n_seen += 1
                cp = engine.eval_cp(fen, depth=DEPTH)
                writer.writerow([fen, cp])
                n_written += 1

                # Print progress occasionally
                if n_written % 50 == 0:
                    print(f"written={n_written} last_cp={cp}")

                # Force flush so you always see file growth
                if n_written % FSYNC_EVERY == 0:
                    f.flush()
                    os.fsync(f.fileno())

                if n_written >= MAX_POSITIONS:
                    print("Reached MAX_POSITIONS.")
                    f.flush()
                    os.fsync(f.fileno())
                    engine.close()
                    return

        # Final flush
        f.flush()
        os.fsync(f.fileno())

    engine.close()
    print(f"Done. seen={n_seen} written={n_written}")


if __name__ == "__main__":
    main()
