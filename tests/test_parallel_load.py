from pathlib import Path

import polars as pl
from helmkit import load_in_parallel
from helmkit import load_monomer_library
from rdkit import Chem
from tqdm import tqdm


def test():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "peptides.csv")
    monomer_db = load_monomer_library(str(data_dir / "monomers.sdf"))
    helms = df["HELM"].to_list()

    molecules_parallel = load_in_parallel(helms, monomer_db)
    for m, row in tqdm(
        zip(molecules_parallel, df.iter_rows(named=True)), total=df.height
    ):
        smiles = row["SMILES"]
        inchi1 = Chem.MolToInchi(m.mol)
        other = Chem.MolFromSmiles(smiles)
        inchi2 = Chem.MolToInchi(other)
        assert inchi1 == inchi2


if __name__ == "__main__":
    test()
