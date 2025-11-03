from pathlib import Path

from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


def main():
    monomer_library = load_monomer_library()
    linkers = load_monomer_library(Path(__file__).parent / "data" / "puromycin.sdf")
    monomer_library["chem"] = linkers["chem"]
    helm = "PEPTIDE1{A.C}|RNA1{R(G)P.R(U)P}|CHEM1{PURO}$PEPTIDE1,CHEM1,2:2-1:1|RNA1,CHEM1,1:1-1:2$$$V2.0"
    molecule = Molecule(helm, monomer_df=monomer_library)
    AllChem.Compute2DCoords(molecule.mol)
    Draw.MolToImage(molecule.mol, size=(800, 800)).show()


if __name__ == "__main__":
    main()
