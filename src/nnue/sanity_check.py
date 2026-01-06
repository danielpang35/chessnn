from nnue.features import extract_features_from_fen

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
pf = extract_features_from_fen(fen)

print("w_king_sq:", pf.w_king_sq)
print("b_king_sq:", pf.b_king_sq)
print("stm:", pf.stm)
print("num pieces:", len(pf.pieces))

fw = pf.features_for_white_king()
fb = pf.features_for_black_king()
print("first 5 features (white king):", fw[:5])
print("first 5 features (black king):", fb[:5])
print("min/max fw:", min(fw), max(fw))
print("min/max fb:", min(fb), max(fb))
