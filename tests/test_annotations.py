import pytest
from helmkit import Molecule
from rdkit import Chem

# HELM2 lets a quoted annotation follow a polymer, a monomer or a connection.
# It is a note for the reader and says nothing about the structure, and its
# text may contain any of the separators HELM is split on.


def smiles(helm):
    return Chem.MolToSmiles(Molecule(helm).mol)


@pytest.mark.parametrize(
    ("annotated", "plain"),
    [
        ('PEPTIDE1{A.C"mutation".G}$$$$', "PEPTIDE1{A.C.G}$$$$"),
        ('PEPTIDE1{A.C"a.b|c$d,e(f]g{h}".G}$$$$', "PEPTIDE1{A.C.G}$$$$"),
        ('PEPTIDE1{A.G""}$$$$', "PEPTIDE1{A.G}$$$$"),
        (
            'PEPTIDE1{A.[C[C@H](N[*])C([*])=O |$;;;_R1;;_R2;$|]"inline".G}$$$$',
            "PEPTIDE1{A.A.G}$$$$",
        ),
        ('RNA1{R(A)P"5\' end".R(C)}$$$$', "RNA1{R(A)P.R(C)}$$$$"),
        (
            'PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3"S-S, bridge|$"$$$',
            "PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$",
        ),
        ('PEPTIDE1{A}"a|b$c"|PEPTIDE2{G}$$$$', "PEPTIDE1{A}|PEPTIDE2{G}$$$$"),
    ],
)
def test_an_annotation_leaves_the_structure_alone(annotated, plain):
    assert smiles(annotated) == smiles(plain)


def test_an_annotated_hydrogen_bond_is_recorded():
    for helm in [
        'RNA1{R(A)}|RNA2{R(U)}$RNA1,RNA2,2:pair-2:pair"Watson-Crick"$$$',
        'RNA1{R(A)}|RNA2{R(U)}$$RNA1,RNA2,2:pair-2:pair"Watson-Crick"$$',
    ]:
        assert Molecule(helm).hydrogen_bonds == [["RNA1", 1, "RNA2", 1]]


def test_an_annotated_connection_still_makes_its_bond():
    """The annotation used to end up in the R-group and the bond was refused."""
    molecule = Molecule('PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3"S-S"$$$')

    assert len(molecule.bond_indices) == 4


def test_the_unused_sections_are_not_read():
    """A malformed extended annotation does not matter to the structure."""
    assert smiles('PEPTIDE1{A}$$${"Name”:"x"}$V2.0') == smiles("PEPTIDE1{A}$$$$")


@pytest.mark.parametrize(
    "helm",
    [
        'PEPTIDE1{A.C"mutation.G}$$$$',
        'PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R1-2:R2"open$$$',
        # An annotation sits after a whole RNA residue, not inside a branch.
        'RNA1{R(A)P.R(U"mutation")P}$$$$',
        # Nor between the monomers of one residue.
        'RNA1{R(A)P.R(U)"mutation"P}$$$$',
        # An annotation alone is not a monomer.
        'PEPTIDE1{A."note".G}$$$$',
    ],
)
def test_a_misplaced_or_unclosed_annotation_is_rejected(helm):
    with pytest.raises(ValueError):
        Molecule(helm)
