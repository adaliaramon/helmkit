from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tqdm import tqdm


def main():
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
        if inchi1 != inchi2:
            mistmatches_dir = Path(__file__).parent / "mismatches"
            mistmatches_dir.mkdir(exist_ok=True)
            AllChem.Compute2DCoords(m.mol, clearConfs=True)
            Chem.AssignAtomChiralTagsFromStructure(m.mol)
            AllChem.Compute2DCoords(other, clearConfs=True)
            Chem.AssignAtomChiralTagsFromStructure(other)
            # Draw the two molecules to an image
            img = Draw.MolsToGridImage(
                [m.mol, other], molsPerRow=2, subImgSize=(800, 800)
            )
            img.save(mistmatches_dir / f"{row['ID']}.png")
            raise ValueError(f"{inchi1} != {inchi2} for row {row}")


if __name__ == "__main__":
    main()
