import random

from rdkit import Chem
from tqdm import tqdm

USE_PYPEPT = False

if USE_PYPEPT:
    from pyPept.converter import Converter
    from pyPept.molecule import Molecule
    from pyPept.sequence import Sequence
else:
    from helmkit import Molecule


# pypept_failures = ['PEPTIDE1{E.E.V.M.I.Y.H.Y.V.R.F.Y.C.V.Y.E.F.D.M.C.F.S.T.T.W.H.F.N.H.Q.G.C.R.T.N.T.D.S.L.G.W.A.M}$$$$V2.0', 'PEPTIDE1{Y.R.W.A.I.M.N.A.V.D.D.C.V.F.P.V.Q.Q.T.C.Q.K.R.Q.H.N.S.W.V.E.K.W.N.C.F.G.R.K.K.S.C.I.Y.T.S.D.D.A.A}$$$$V2.0', 'PEPTIDE1{Y.W.T.M.H.R.P.Q.N.M.I.V.C.P.P.N.E.K.L.Q.L.C.A.N.R.E.H.S.L.D.S.K.D.S.Y.H.R.Q.M.Q.L.M.D.Y.E.K.D.I.F.I}$$$$V2.0', 'PEPTIDE1{N.N.S.C.Q.N.N.H.I.W.N.H.A.D.A.G.N.H.N.M.F.R.L.L.L.Y.T.F.H.F.H.I.F.M.T.P.N.K.A.A.M.Q.Q.N.K.T.V.S.A.N}$$$$V2.0', 'PEPTIDE1{S.T.D.W.H.Q.K.H.D.Y.N.H.S.S.G.G.R.H.L.C.L.N.W.F.W.H.A.W.N.A.C.V.E.S.D.C.C.N.L.M.K.L.K.I.Y.Q.P.V.H.H}$$$$V2.0', 'PEPTIDE1{F.N.C.V.C.D.Q.C.K.G.I.K.T.V.I.H.I.S.Q.F.D.T.T.K.L.V.L.M.P.T.V.T.T.D.M.T.M.K.I.K.Y.F.Y}$$$$V2.0', 'PEPTIDE1{N.T.L.D.I.R.V.W.S.P.D.R.A.D.V.V.M.L.M.V.A.A.F.S.S.R.F.D.A.V.F.S.T.A.F.L.E.I.H.S.Q}$$$$V2.0', 'PEPTIDE1{A.N.W.W.F.T.F.H.T.W.S.I.L.V.I.S.V.A.V.M.I.K.Y.R.H.V.L.W.F.C.Y.Y.G.N.R.S.V.H}$$$$V2.0']


def test():
    random.seed(0)
    aminoacids = "ACDEFGHIKLMNPQRSTVWY"
    errors = []
    for _ in tqdm(range(1000)):
        num_monomers = random.randint(2, 50)
        peptide = "".join(random.choices(aminoacids, k=num_monomers))
        helm = f"PEPTIDE1{{{'.'.join(peptide)}}}$$$$V2.0"
        try:
            if USE_PYPEPT:
                converter = Converter(helm=helm)
                sequence = Sequence(converter.get_biln())
                molecule = Molecule(sequence)
            else:
                molecule = Molecule(helm)
            inchi1 = Chem.MolToInchi(molecule.mol)
            inchi2 = Chem.MolToInchi(Chem.MolFromSequence(peptide))
            assert inchi1 == inchi2, (helm, inchi1, inchi2)
        except Exception:
            errors.append(helm)
            raise
    print(f"Success rate: {(1000 - len(errors)) / 1000 * 100:.2f}%")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    test()
