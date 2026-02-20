from pathlib import Path

from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


def main():
    monomer_library = load_monomer_library()
    linkers = load_monomer_library(Path(__file__).parent / "data" / "linkers.sdf")
    monomer_library["chem"] = linkers["chem"]
    helm = "PEPTIDE1{A.C}|RNA1{R(G)P.R(U)}|CHEM1{PURO}$PEPTIDE1,CHEM1,2:R2-1:R2|RNA1,CHEM1,1:R1-1:R1$$$V2.0"
    molecule = Molecule(helm, monomer_df=monomer_library)
    # AllChem.Compute2DCoords(molecule.mol)
    # Draw.MolToImage(molecule.mol, size=(800, 800)).show()

    helm = "PEPTIDE1{A.L}|RNA1{p.r(U)}|CHEM1{PAD4}$PEPTIDE1,CHEM1,2:R2-1:R1|RNA1,CHEM1,1:R1-1:R2$${}$V2.0"
    molecule = Molecule(helm, monomer_df=monomer_library)
    AllChem.Compute2DCoords(molecule.mol)
    Draw.MolToImage(
        molecule.mol, size=(800, 800), highlightBonds=molecule.get_broken_bond_idx()
    ).show()
    raise SystemExit(0)

    helm = "PEPTIDE1{C.A.L}|RNA1{p.r(U)}|CHEM1{FAKE}$PEPTIDE1,CHEM1,1:R3-1:R1|RNA1,CHEM1,1:R1-1:R2$${}$V2.0"
    molecule = Molecule(helm, monomer_df=monomer_library)
    AllChem.Compute2DCoords(molecule.mol)
    Draw.MolToImage(molecule.mol, size=(800, 800)).show()


if __name__ == "__main__":
    main()
