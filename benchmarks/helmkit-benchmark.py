import argparse
import time
from pathlib import Path

import polars as pl
from helmkit import load_in_parallel
from helmkit import load_monomer_library
from helmkit import Molecule


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--reload", action="store_true", help="reload the library for every peptide"
    )
    mode.add_argument(
        "--parallel", action="store_true", help="parse with load_in_parallel"
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "peptides.csv")

    # Remove peptides with monomers containing parentheses, spaces or hyphens
    # (as these do not work with pyPept)
    regex = r"\[[^\]]*[\(\s-][^\]]*\]"
    df = df.filter(pl.col("HELM").str.contains(regex).not_())
    helms = df["HELM"].to_list()

    library_path = data_dir / "monomers.sdf"
    # Bypass the cache when reloading so every peptide pays for loading the library
    load_library = (
        load_monomer_library.__wrapped__ if args.reload else load_monomer_library
    )
    monomer_db = load_library(library_path)

    start = time.perf_counter()
    if args.parallel:
        load_in_parallel(helms, monomer_db)
    else:
        for helm in helms:
            if args.reload:
                monomer_db = load_library(library_path)
            Molecule(helm, monomer_db)
    end = time.perf_counter()
    print(f"Processed {df.height} peptides in {end - start:.2f} seconds")
    print(f"Average time per peptide: {(end - start) / df.height:.6f} seconds")


if __name__ == "__main__":
    main()
