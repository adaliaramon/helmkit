from pathlib import Path

import polars as pl
from helmkit import Molecule
from rdkit import Chem
from tqdm import tqdm

USE_PYPEPT = False

if USE_PYPEPT:
    from pyPept.converter import Converter
    from pyPept.molecule import Molecule
    from pyPept.sequence import Sequence
else:
    from helmkit import Molecule


def test():
    data_dir = Path(__file__).parent / "data"
    df = pl.read_csv(data_dir / "peptides.csv")

    if USE_PYPEPT:
        monomer_lib_dir = str(data_dir.relative_to(Path.cwd()))
        monomer_lib = "monomers.sdf"

    errors = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        helm = row["HELM"]
        smiles = row["SMILES"]
        try:
            if USE_PYPEPT:
                converter = Converter(helm=helm)
                sequence = Sequence(converter.get_biln(), monomer_lib_dir, monomer_lib)
                m = Molecule(sequence)
            else:
                m = Molecule(helm)
            inchi1 = Chem.MolToInchi(m.mol)
            other = Chem.MolFromSmiles(smiles)
            inchi2 = Chem.MolToInchi(other)
            assert inchi1 == inchi2
        except (Exception, SystemExit):
            errors.append(helm)
            raise
    print(f"Success rate: {(df.height - len(errors)) / df.height * 100:.2f}%")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    test()
