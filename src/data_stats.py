"""Prints per-split, per-language corpus stats: line count, blank lines,
word-length percentiles, and the first few lines (to eyeball encoding).
Asserts .en and .vi have the same number of lines in every split.
"""
import argparse
import glob
import os

import numpy as np


def stats_for(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    lengths = np.array([len(line.split()) for line in lines])
    p50, p90, p99 = np.percentile(lengths, [50, 90, 99])
    return {
        "n_lines": len(lines),
        "n_blank": sum(1 for line in lines if not line.strip()),
        "p50": p50, "p90": p90, "p99": p99, "max": int(lengths.max()),
        "head": lines[:3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    splits = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(args.dir, "*.en")))

    for split in splits:
        en = stats_for(os.path.join(args.dir, f"{split}.en"))
        vi = stats_for(os.path.join(args.dir, f"{split}.vi"))
        assert en["n_lines"] == vi["n_lines"], (
            f"{split}: line count mismatch, en={en['n_lines']} vi={vi['n_lines']}"
        )

        print(f"=== {split} ({en['n_lines']} lines) ===")
        for lang, s in [("en", en), ("vi", vi)]:
            print(f"  {lang}: blank={s['n_blank']}  words/line p50={s['p50']:.0f} "
                  f"p90={s['p90']:.0f} p99={s['p99']:.0f} max={s['max']}")
            for line in s["head"]:
                print(f"    | {line}")
        print()


if __name__ == "__main__":
    main()
