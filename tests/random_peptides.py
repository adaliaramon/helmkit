import random

from helmkit import Molecule
from rdkit import Chem
from tqdm import tqdm


def main():
    random.seed(0)
    aminoacids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in tqdm(range(1000)):
        num_monomers = random.randint(2, 50)
        peptide = "".join(random.choices(aminoacids, k=num_monomers))
        helm = f"PEPTIDE1{{{'.'.join(peptide)}}}$$$$V2.0"
        molecule = Molecule(helm)
        inchi1 = Chem.MolToInchi(molecule.mol)
        inchi2 = Chem.MolToInchi(Chem.MolFromSequence(peptide))
        if inchi1 != inchi2:
            raise ValueError(f"{inchi1} != {inchi2}")


if __name__ == "__main__":
    main()
