# chess-nnue

## Stockfish dataset generator CLI

The Stockfish generator script lives at `src/nnue/stockfish.py`. Run it from the
repo root, overriding any defaults with the flags below:

```
python src/nnue/stockfish.py [options]
```

### Flags

| Flag | Purpose |
| --- | --- |
| `--stockfish PATH` | Path to the Stockfish executable. |
| `--pgn PATH` | PGN source: a single file or a directory of `.pgn` files (searched non-recursively). |
| `--out PATH` | Output CSV path. Parent directories are created automatically. |
| `--append` | Resume/appends to an existing CSV. Counts current rows, writes more, and stops at the new target total. |
| `--max-positions N` | Number of *new* rows to write per invocation (or until combined total is reached when `--append` is set). |
| `--depth N` | Stockfish search depth used for labeling (default search limit). |
| `--nodes N` | Search until N nodes are visited (mutually exclusive with `--depth`/`--movetime`). |
| `--movetime MS` | Search for MS milliseconds per position (mutually exclusive with `--depth`/`--nodes`). |
| `--mate-cp N` | Centipawn clamp for mate scores (mates become ±N). |
| `--threads N` | `Threads` engine option (parallel search; defaults to detected core count). |
| `--hash-mb N` | `Hash` engine option in MB. |
| `--min-ply N` / `--max-ply N` | Only sample positions between these ply bounds. |
| `--sample-every N` | Sample every N plies within the allowed window. |
| `--report-every N` | Print progress every N written rows. |
| `--fsync-every N` | Flush and fsync the CSV every N rows (set to 0 to disable extra flushing). |

### Common workflows

- **Fresh run:**
  ```
  python src/nnue/stockfish.py --stockfish /path/to/stockfish --pgn data/raw/games.pgn --out data/raw/positions.csv --max-positions 200000
  ```

- **Append/resume toward a larger goal (no need to clear the file):**
  ```
  python src/nnue/stockfish.py --append --out data/raw/positions.csv --max-positions 150000
  ```
  The script counts existing rows and stops when `existing + max_positions` is reached.

- **Sweep a folder of PGNs:**
  ```
  python src/nnue/stockfish.py --pgn /data/pgns --out data/raw/positions.csv --append
  ```

- **Adjust sampling or search cost:**
  ```
  python src/nnue/stockfish.py --sample-every 4 --min-ply 6 --max-ply 120 --depth 12
  ```

- **Reduce disk sync overhead for big batches:** set `--fsync-every 0` (or a larger number) once the workflow is stable.

- **High-throughput labeling:** swap fixed depth for a per-position time cap and enable more threads:
  ```
  python src/nnue/stockfish.py --movetime 75 --threads 8 --hash-mb 512 --report-every 200
  ```
  This yields many more positions per hour than a deep fixed depth while still producing usable targets. Increase `--movetime` for stronger labels or decrease for speed.
