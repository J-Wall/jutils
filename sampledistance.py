#!/usr/bin/env python

from collections import defaultdict
from multiprocessing import Pool
from os import PathLike
from typing import Optional

from cyvcf2 import VCF
from numba import njit
import numpy as np
from tqdm import tqdm


@njit
def cityblock(a: np.ndarray, b: np.ndarray) -> int:
    return np.sum(np.abs(a - b))


@njit
def genotype_distance(genotype_a, genotype_b) -> int:
    if np.any(genotype_a < 0) or np.any(genotype_b < 0):
        return 0

    minlength = max(np.max(genotype_a), np.max(genotype_b)) + 1
    return cityblock(
        np.bincount(genotype_a, minlength=minlength),
        np.bincount(genotype_b, minlength=minlength),
    )


@njit
def pdist(genotypes: np.ndarray) -> np.ndarray:
    n = len(genotypes)
    dist = np.zeros((n, n), dtype="i4")

    for i in range(n):
        for j in range(i, n):
            dist[i, j] = genotype_distance(genotypes[i, :], genotypes[j, :])

    return dist


@njit
def joint_called(called: np.ndarray) -> np.ndarray:
    n = called.shape[0]
    joint = np.zeros((n, n), dtype="b")

    for i in range(n):
        for j in range(i, n):
            joint[i, j] = called[i] & called[j]

    return joint


def _pdist_and_n_called(
    genotypes: tuple[str, np.ndarray]
) -> tuple[str, np.ndarray, np.ndarray]:
    dist = pdist(genotypes[1])
    n_called = joint_called(genotypes[1][:, 0] >= 0)

    return genotypes[0], dist, n_called


SampleMatrices = dict[str, tuple[np.ndarray, np.ndarray, int]]


def get_sample_matrices(
    vcf: VCF,
    region: Optional[str] = None,
    processes: Optional[int] = None,
    chunksize: int = 1,
) -> SampleMatrices:
    n = len(vcf.samples)
    _dist = np.zeros((n, n), dtype="i4")
    _n_called = np.zeros((n, n), dtype="i4")

    d = defaultdict(lambda: [_dist.copy(), _n_called.copy(), 0])

    genotypeses = ((var.CHROM, np.array(var.genotypes)[:, :-1]) for var in vcf(region))

    with Pool(processes) as pool:
        for chrom, dist, n_called in tqdm(
            pool.imap_unordered(_pdist_and_n_called, genotypeses, chunksize),
            unit="variants",
        ):
            d[chrom][0] += dist
            d[chrom][1] += n_called
            d[chrom][2] += 1

    return {k: tuple(v) for k, v in d.items()}


def save_sample_matrices(f: PathLike, d: SampleMatrices) -> None:
    save_dict = {}
    for chrom, (dist, n_called, n_variants) in d.items():
        save_dict[f"distance__{chrom}"] = dist
        save_dict[f"n_called__{chrom}"] = n_called
        save_dict[f"n_variants__{chrom}"] = n_variants

    np.savez_compressed(f, **save_dict)


def load_sample_matrices(f: PathLike) -> SampleMatrices:
    d = {}

    identifiers = ["distance__", "n_called__", "n_variants__"]

    with np.load(f) as data:
        for k in data.keys():
            if k.startswith(identifiers[0]):
                chrom = k[len(identifiers[0]) :]
                l = []
                for identifier in identifiers:
                    l.append(data[f"{identifier}{chrom}"])

                d[chrom] = tuple(l)

    return d


if __name__ == "__main__":
    import argparse
    from sys import stderr

    parser = argparse.ArgumentParser(
        description="Calculate manhattan distance between samples in a BCF/VCF file.",
    )

    parser.add_argument("bcf", help="BCF/VCF file to process.")
    parser.add_argument("out", help="Output .npz file.")
    parser.add_argument(
        "-r",
        "--region",
        type=str,
        default=None,
        help="Restrict processing to a specific region.",
    )
    parser.add_argument(
        "-t",
        "--parser-threads",
        type=int,
        default=None,
        help="Number of threads used for parsing BCF/VCF file.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="Number of processes to spawn for computing distances.",
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=20,
        help="Chunk size for multiprocessing queue.",
    )

    args = parser.parse_args()

    vcf = VCF(args.bcf, lazy=True, threads=args.parser_threads)
    d = get_sample_matrices(
        vcf, region=args.region, processes=args.processes, chunksize=args.chunksize
    )

    save_sample_matrices(args.out, d)
    print(f"Sample matrices saved to {args.out}.", file=stderr)
