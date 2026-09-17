import re
from pathlib import Path

import polars as pl
from helmkit import load_monomer_library
from helmkit import Molecule
from helmkit.molecule import _create_missing_monomer
from rdkit import Chem
from tqdm import tqdm


def states_double_bond_geometry(helm: str) -> bool:
    r"""Does the HELM string say which way round a double bond goes?

    Only an inline SMILES monomer can, and only with a / or a \.
    """
    return bool(re.search(r"[/\\]", helm))


def clean_inchi(inchi: str, compare_double_bond_stereo: bool = False) -> str:
    # Remove /b... (double bond stereo) layer, unless the HELM string says what
    # the geometry is, in which case helmkit is expected to reproduce it.
    if not compare_double_bond_stereo:
        inchi = re.sub(r"/b[^/]+", "", inchi)
    # Remove /p... (charge) layer. It records the protonation state of the
    # structure PubChem happens to hold, which a HELM string does not describe:
    # the references that differ here are recorded as salts of up to +8 while
    # the HELM describes the neutral molecule.
    return re.sub(r"/p[+-]?\d*", "", inchi)


def test():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_ndjson(data_dir / "pubchem.ndjson")
    monomer_db = load_monomer_library()

    monomer_db["aa"]["Glp"] = _create_missing_monomer(
        "O=C1N[C@@H](CC1)C(=O)* |$;;;;;;;;_R2$|"
    )
    ggu = _create_missing_monomer("*N[C@@H](CCC(=O)*)C(=O)* |$_R1;;;;;;;_R2;;;_R3$|")
    ggu["m_Rgroups"][2] = "OH"
    monomer_db["aa"]["Ggu"] = ggu
    monomer_db["aa"]["Tml"] = _create_missing_monomer(
        "*N[C@@H](CCCC[N+](C)(C)C)C(=O)* |$_R1;;;;;;;;;;;;;_R2$|"
    )
    monomer_db["aa"]["Dpr"] = _create_missing_monomer(
        "*N[C@@H](CN*)C(=O)* |$_R1;;;;;_R3;;;_R2$|"
    )
    monomer_db["aa"]["Har"] = _create_missing_monomer(
        "C(CCN=C(N)N)C[C@@H](C(=O)*)N* |$;;;;;;;;;;;_R2;;_R1$|"
    )

    # Monomers with incorrect formula (extra OH which should be an R-group)
    monomer_db["aa"][
        "*C(=O)(CC[C@@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCC)O |$_R3;;;;;;;;;;;;;;;;;;;;;;;;;;;$|"
    ] = _create_missing_monomer(
        "*C(=O)(CC[C@@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCC) |$_R3;;;;;;;;;;;;;;;;;;;;;;;;;;;$|"
    )
    monomer_db["aa"][
        "*C(=O)(CCC(C(=O)O)NC(=O)CCCCCCCCCCCCCCC)O |$_R3;;;;;;;;;;;;;;;;;;;;;;;;;;;$|"
    ] = _create_missing_monomer(
        "*C(=O)(CCC(C(=O)O)NC(=O)CCCCCCCCCCCCCCC) |$_R3;;;;;;;;;;;;;;;;;;;;;;;;;;;$|"
    )
    monomer_db["aa"]["*C(=O)(CCCC(C(=O)O)N)O |$_R3;;;;;;;;;;;$|"] = (
        _create_missing_monomer("*C(=O)(CCCC(C(=O)O)N) |$_R3;;;;;;;;;;;$|")
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
            raise
            continue
        inchi1 = Chem.MolToInchi(m.mol)
        if m.has_ambiguous_monomers:
            # Ignore stereo
            inchi1 = inchi1.split("/")[0]
            inchi2 = inchi2.split("/")[0]
        compare_stereo = states_double_bond_geometry(helm)
        inchi1 = clean_inchi(inchi1, compare_stereo)
        inchi2 = clean_inchi(inchi2, compare_stereo)
        if inchi1 != inchi2:
            errors.append(row)
            reasons.append(f"{inchi1} != {inchi2}")
    for error, reason in zip(errors, reasons):
        print("-" * 100)
        print(error)
        print(reason)
        print("-" * 100)
    print(f"Found {len(errors)} errors")
    if len(errors) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test()
