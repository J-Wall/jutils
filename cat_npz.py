#!/usr/bin/env python

import numpy as np
from os import PathLike


def merge_npz(files: list[PathLike]) -> dict[str, np.ndarray]:
    d = {}
    for f in files:
        with np.load(f) as data:
            for k, v in data.items():
                d[k] = v

    return d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merges npz files.")
    parser.add_argument(
        "infile", nargs="+", help="Files to merge. Assumes all keys are unique."
    )
    parser.add_argument("outfile", help="Path to output file.")

    args = parser.parse_args()

    d = merge_npz(args.infile)
    np.savez_compressed(args.outfile, **d)
