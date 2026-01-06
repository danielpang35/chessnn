from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import chess

NUM_SQUARES = 64
NUM_PIECES = 12  # WP..WK, BP..BK


# Frozen mapping:
# 0..5   = white pawn..king
# 6..11  = black pawn..king
PIECE_TO_INDEX = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def halfkp_feature_index(king_sq: int, piece_idx: int, piece_sq: int) -> int:
    """
    Feature index for HalfKP:

      feature = (king_square, piece_type_with_color, piece_square)

    Flattened as:
      ((king_sq * 12 + piece_idx) * 64 + piece_sq)

    Range: [0, 64*12*64)
    """
    return (king_sq * NUM_PIECES + piece_idx) * NUM_SQUARES + piece_sq


@dataclass(frozen=True)
class PositionFeatures:
    w_king_sq: int
    b_king_sq: int
    pieces: List[Tuple[int, int]]  # list of (piece_idx, piece_sq)
    stm: int  # 0 = white, 1 = black

    def features_for_white_king(self) -> List[int]:
        k = self.w_king_sq
        return [halfkp_feature_index(k, p, s) for (p, s) in self.pieces]

    def features_for_black_king(self) -> List[int]:
        k = self.b_king_sq
        return [halfkp_feature_index(k, p, s) for (p, s) in self.pieces]


def extract_features_from_fen(fen: str) -> PositionFeatures:
    board = chess.Board(fen)

    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)
    if w_king_sq is None or b_king_sq is None:
        raise ValueError("FEN missing a king")

    pieces: List[Tuple[int, int]] = []
    for sq, piece in board.piece_map().items():
        idx = PIECE_TO_INDEX[(piece.color, piece.piece_type)]
        pieces.append((idx, sq))

    stm = 0 if board.turn == chess.WHITE else 1

    return PositionFeatures(
        w_king_sq=w_king_sq,
        b_king_sq=b_king_sq,
        pieces=pieces,
        stm=stm,
    )
