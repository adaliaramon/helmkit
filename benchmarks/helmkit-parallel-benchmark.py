import time
from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import load_peptides_in_parallel


def main():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "peptides.csv")

    # Remove peptides with monomers containing parentheses, spaces or hyphens
    # (as these do not work with pyPept)
    regex = r"\[[^\]]*[\(\s-][^\]]*\]"
    df = df.filter(pl.col("HELM").str.contains(regex).not_())
    helms = df["HELM"].to_list()

    monomer_db = load_monomer_library(data_dir / "monomers.sdf")

    start = time.perf_counter()
    load_peptides_in_parallel(helms, monomer_db)
    end = time.perf_counter()
    print(f"Processed {df.height} peptides in {end - start:.2f} seconds")
    print(f"Average time per peptide: {(end - start) / df.height:.6f} seconds")


if __name__ == "__main__":
    main()
