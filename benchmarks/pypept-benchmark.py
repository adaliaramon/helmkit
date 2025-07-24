import time
from pathlib import Path

import polars as pl
from pyPept.converter import Converter
from pyPept.molecule import Molecule
from pyPept.sequence import Sequence


def main():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "peptides.csv")

    # Remove peptides with monomers containing parentheses, spaces or hyphens
    # (as these do not work with pyPept)
    regex = r"\[[^\]]*[\(\s-][^\]]*\]"
    df = df.filter(pl.col("HELM").str.contains(regex).not_())
    helms = df["HELM"].to_list()

    monomer_lib_dir = str(data_dir.relative_to(Path.cwd()))
    monomer_lib = "monomers.sdf"

    start = time.perf_counter()
    for helm in helms:
        converter = Converter(helm=helm)
        sequence = Sequence(converter.get_biln(), monomer_lib_dir, monomer_lib)
        Molecule(sequence)
    end = time.perf_counter()
    print(f"Processed {df.height} peptides in {end - start:.2f} seconds")
    print(f"Average time per peptide: {(end - start) / df.height:.6f} seconds")


if __name__ == "__main__":
    main()
