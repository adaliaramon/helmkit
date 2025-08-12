import re
from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import Molecule
from helmkit.molecule import _create_missing_monomer
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm


def clean_inchi(inchi: str) -> str:
    # Remove /b... (double bond stereo) layer
    inchi = re.sub(r"/b[^/]+", "", inchi)
    # Remove /p... (charge) layer
    inchi = re.sub(r"/p[+-]?\d*", "", inchi)
    return inchi


def main():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "CID-Peptides-filtered.csv")
    monomer_db = load_monomer_library()
    monomer_db_2 = load_monomer_library(data_dir / "monomers.sdf")
    monomer_db.update(monomer_db_2)

    for row in pl.read_csv(data_dir / "extra-monomers.csv").iter_rows(named=True):
        smiles = row["SMILES"]
        monomer = _create_missing_monomer(smiles)
        if row["R-groups"]:
            r_groups = row["R-groups"].split(",")
        else:
            r_groups = ["None", "OH" if "_R2" in smiles else "None", "None", "None"]
        for i, r in enumerate(r_groups):
            if r != "None":
                monomer["m_Rgroups"][i] = r
        monomer_db[row["Name"]] = monomer

    errors = pl.read_csv(data_dir / "errors.csv")
    df = df.filter(pl.col("CID").is_in(errors["CID"].implode()).not_())

    for row in tqdm(df.iter_rows(named=True), total=df.height):
        helm = row["HELM"]
        mol2 = Chem.MolFromSmiles(row["SMILES"])
        # Skip molecules that contain isotopes
        if any(atom.GetIsotope() != 0 for atom in mol2.GetAtoms()):
            continue
        inchi2 = row["InChI"]
        try:
            m = Molecule(helm, monomer_db)
        except:
            print(row)
            raise
        with rdBase.BlockLogs():
            inchi1 = Chem.MolToInchi(m.mol)
        if inchi1 == "":
            # RDKit bug with InsertMol
            continue
        if m.has_ambiguous_monomers:
            # Ignore stereo
            inchi1 = inchi1.split("/")[0]
            inchi2 = inchi2.split("/")[0]
        inchi1 = clean_inchi(inchi1)
        inchi2 = clean_inchi(inchi2)
        assert inchi1 == inchi2, (
            row,
            inchi1,
            inchi2,
            m.monomers,
            m.bondlist,
            m.offset,
            # Chem.MolToMolBlock(m.mol),
        )


if __name__ == "__main__":
    main()
