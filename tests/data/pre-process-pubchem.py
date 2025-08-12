from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm


def main():
    data_dir = Path(__file__).parent
    df = pl.read_csv(data_dir / "CID-Peptides.csv")
    errors = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        smiles = row["HELM"][10:-6]
        if "." in smiles:
            continue
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            inchi = Chem.MolToInchi(mol)
        if inchi != row["InChI"]:
            errors.append(row["CID"])
    df.filter(pl.col("CID").is_in(errors).not_()).write_csv(
        data_dir / "CID-Peptides-filtered.csv"
    )


if __name__ == "__main__":
    main()
