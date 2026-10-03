from pathlib import Path

import polars as pl
from rdkit import Chem
from tqdm import tqdm

from helmkit import load_in_parallel
from helmkit import load_monomer_library


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


def test_molecules_share_the_callers_library():
    # Workers do not send the library back with their results, so every
    # molecule has to be handed the caller's own one again rather than a copy.
    monomer_db = load_monomer_library()
    molecules = load_in_parallel(
        ["PEPTIDE1{A.G}$$$$", "PEPTIDE1{L}$$$$"], monomer_db, chunksize=1
    )
    assert all(m.monomer_df is monomer_db for m in molecules)


if __name__ == "__main__":
    test()
