from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit import Chem
from tqdm import tqdm


def test():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "peptides.csv")
    monomer_db = load_monomer_library(data_dir / "monomers.sdf")
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        helm = row["HELM"]
        smiles = row["SMILES"]
        try:
            m = Molecule(helm, monomer_db)
        except:
            print(row)
            raise
        inchi1 = Chem.MolToInchi(m.mol)
        other = Chem.MolFromSmiles(smiles)
        inchi2 = Chem.MolToInchi(other)
        assert inchi1 == inchi2


if __name__ == "__main__":
    test()
