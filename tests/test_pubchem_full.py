import re
from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import Molecule
from helmkit.molecule import _create_missing_monomer
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

h_number_pattern = re.compile(r"H(\d+)")


def clean_inchi(inchi: str) -> str:
    # Remove /b... (double bond stereo) layer
    inchi = re.sub(r"/b[^/]+", "", inchi)
    # Remove /p... (charge) layer
    inchi = re.sub(r"/p[+-]?\d*", "", inchi)
    return inchi


def count_hydrogens(inchi: str) -> int:
    return int(h_number_pattern.search(inchi).group(1))


def main():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "CID-Peptides-filtered-2.csv")
    monomer_db = load_monomer_library()
    monomer_db_2 = load_monomer_library(data_dir / "monomers.sdf")
    monomer_db["aa"].update(monomer_db_2["aa"])

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

    matches = 0
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        # TODO: remove
        # if row["CID"] < 10919436:
        #     continue
        helm = row["HELM"]
        mol2 = Chem.MolFromSmiles(row["SMILES"])
        # Skip molecules that contain isotopes
        if any(atom.GetIsotope() != 0 for atom in mol2.GetAtoms()):
            continue
        inchi2 = row["InChI"]
        try:
            m = Molecule(helm, monomer_db)
        except:
            # continue
            print(row)
            raise
        with rdBase.BlockLogs():
            inchi1 = Chem.MolToInchi(m.mol)
        if inchi1 == "":
            # RDKit bug with InsertMol
            continue
        inchi1 = clean_inchi(inchi1)
        inchi2 = clean_inchi(inchi2)
        if m.has_ambiguous_monomers:
            # Ignore stereo
            stereo_marker = "/t"
            if stereo_marker in inchi1:
                inchi1 = inchi1[: inchi1.index(stereo_marker)]
            if stereo_marker in inchi2:
                inchi2 = inchi2[: inchi2.index(stereo_marker)]
        if count_hydrogens(inchi1) - count_hydrogens(inchi2) == 2:
            # Missing ring in monomer most likely
            continue
        if inchi1 == inchi2:
            matches += 1
        continue
        assert inchi1 == inchi2, (
            row,
            inchi1,
            inchi2,
            m.monomers,
            m.bondlist,
            m.offset,
            # Chem.MolToMolBlock(m.mol),
            matches,
        )
    print(f"Success rate: {100 * matches / df.height:.2f}%")


if __name__ == "__main__":
    main()
