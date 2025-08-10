import re
from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import Molecule
from helmkit.molecule import _create_missing_monomer
from rdkit import Chem
from tqdm import tqdm


def clean_inchi(inchi: str) -> str:
    # Remove /b... (double bond stereo) layer
    inchi = re.sub(r"/b[^/]+", "", inchi)
    # Remove /p... (charge) layer
    inchi = re.sub(r"/p[+-]?\d*", "", inchi)
    return inchi


def test():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_ndjson(data_dir / "pubchem.ndjson")
    monomer_db = load_monomer_library()
    monomer_db_2 = load_monomer_library(data_dir / "monomers.sdf")
    monomer_db.update(monomer_db_2)

    monomer_db["Glp"] = _create_missing_monomer(
        "O=C1N[C@@H](CC1)C(=O)* |$;;;;;;;;_R2$|"
    )
    ggu = _create_missing_monomer("*N[C@@H](CCC(=O)*)C(=O)* |$_R1;;;;;;;_R2;;;_R3$|")
    ggu["m_Rgroups"][2] = "OH"
    monomer_db["Ggu"] = ggu
    monomer_db["Tml"] = _create_missing_monomer(
        "*N[C@@H](CCCC[N+](C)(C)C)C(=O)* |$_R1;;;;;;;;;;;;;_R2$|"
    )
    monomer_db["Dpr"] = _create_missing_monomer(
        "*N[C@@H](CN*)C(=O)* |$_R1;;;;;_R3;;;_R2$|"
    )

    errors = []
    reasons = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        helm = row["HELM"]
        mol2 = Chem.MolFromSmiles(row["SMILES"])
        # Skip molecules that contain isotopes
        if any(atom.GetIsotope() != 0 for atom in mol2.GetAtoms()):
            continue
        inchi2 = row["InChI"]
        try:
            m = Molecule(helm, monomer_db)
        except Exception as e:
            errors.append(row)
            reasons.append(e)
            continue
        inchi1 = Chem.MolToInchi(m.mol)
        if m.has_ambiguous_monomers:
            # Ignore stereo
            inchi1 = inchi1.split("/")[0]
            inchi2 = inchi2.split("/")[0]
        inchi1 = clean_inchi(inchi1)
        inchi2 = clean_inchi(inchi2)
        if inchi1 != inchi2:
            errors.append(row)
            reasons.append(f"{inchi1} != {inchi2}")
        # assert inchi1 == inchi2, (
        #     inchi1,
        #     inchi2,
        #     helm,
        #     row["SMILES"],
        #     draw(m.mol, Chem.MolFromSmiles(row["SMILES"])),
        # )
    for error, reason in zip(errors, reasons):
        if not isinstance(reason, str):
            continue
        print("-" * 100)
        print(error)
        print(reason)
        print("-" * 100)
    print(f"Found {len(errors)} errors")
    if len(errors) > 0:
        exit(1)


if __name__ == "__main__":
    test()
