import random

from helmkit import Molecule
from rdkit import Chem
from tqdm import tqdm


def test():
    random.seed(0)
    monomers = "ACGU"
    for _ in tqdm(range(1000)):
        num_monomers = random.randint(2, 40)
        rna = "".join(random.choices(monomers, k=num_monomers))
        residues = [f"r({m})p" for m in rna]
        helm = f"RNA1{{{'.'.join(residues)[:-1]}}}$$$$V2.0"
        molecule = Molecule(helm)
        inchi1 = Chem.MolToInchi(molecule.mol)
        inchi2 = Chem.MolToInchi(Chem.MolFromSequence(rna, flavor=2))
        assert inchi1 == inchi2, (helm, inchi1, inchi2)


if __name__ == "__main__":
    test()
