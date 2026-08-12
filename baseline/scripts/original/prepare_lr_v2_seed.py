"""
Prepare the shared seed directory for the lr_v2 experiments.

What this does:
  1. Copies checkpoint_100.pth.tar from the donor run as best.pth.tar
  2. Loads latest.examples from the donor run, trims to the first 100
     iterations of history, re-saves with iteration=100

Usage (run on the cluster, inside quoridor-ml-bot/):
  python3 scripts/prepare_lr_v2_seed.py [--donor DIR] [--out DIR]

Defaults:
  --donor  external/alphazero/training_runs/acceleration_experiments_lr_v1/bot1_lr_halve200
  --out    external/alphazero/training_runs/lr_v2_seed
"""

import argparse
import pickle
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--donor",
        type=Path,
        default=Path(
            "external/alphazero/training_runs/"
            "acceleration_experiments_lr_v1/bot1_lr_halve200"
        ),
        help="Source experiment folder (must contain checkpoint_100.pth.tar "
             "and latest.examples)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("external/alphazero/training_runs/lr_v2_seed"),
        help="Destination seed directory",
    )
    p.add_argument(
        "--keep-iters",
        type=int,
        default=100,
        help="Number of per-iteration history entries to keep (default 100)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    donor: Path = args.donor
    out: Path = args.out
    keep = args.keep_iters

    # --- Validate donor ---
    ckpt_src = donor / "checkpoint_100.pth.tar"
    examples_src = donor / "latest.examples"

    if not ckpt_src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_src}")
    if not examples_src.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_src}")

    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Copy checkpoint_100 as best.pth.tar ---
    best_dst = out / "best.pth.tar"
    shutil.copy2(ckpt_src, best_dst)
    print(f"Copied {ckpt_src} -> {best_dst}")

    # --- 2. Trim examples ---
    with open(examples_src, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict) and "examples" in loaded:
        history = loaded["examples"]
        donor_iter = loaded.get("iteration", "?")
    else:
        # old flat-list format
        history = loaded
        donor_iter = "?"

    total = len(history)
    print(f"Donor examples: {total} iterations recorded (donor iteration marker: {donor_iter})")

    if total < keep:
        raise ValueError(
            f"Donor only has {total} iterations of examples, need at least {keep}"
        )

    trimmed = history[:keep]
    print(f"Trimmed to {len(trimmed)} iterations")

    data_to_save = {"iteration": keep, "examples": trimmed}
    examples_dst = out / "latest.examples"
    with open(examples_dst, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Saved trimmed examples -> {examples_dst}")

    print(f"\nSeed directory ready: {out}")
    print("Contents:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
