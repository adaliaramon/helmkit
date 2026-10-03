import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from helmkit import Molecule

# HELM2 inline SMILES may mark R-group n with atom map n on the atom that leaves
# when the bond is made. Those atoms used to be kept as ordinary atoms, so a
# bond through them overfilled the atom they hung from.


def smiles(helm):
    return Chem.MolToSmiles(Molecule(helm).mol)


@pytest.mark.parametrize(
    "monomer",
    [
        "[C[C@H](N[H:1])C([OH:2])=O]",
        "[C[C@H](N[*:1])C([*:2])=O]",
        "[C[C@H](N[*])C([*])=O |$;;;_R1;;_R2;$|]",
    ],
)
def test_every_spelling_of_an_inline_alanine_is_alanine(monomer):
    expected = smiles("PEPTIDE1{G.A.G}$$$$")

    assert smiles(f"PEPTIDE1{{G.{monomer}.G}}$$$$") == expected
    assert smiles(f"PEPTIDE1{{{monomer}}}$$$$") == smiles("PEPTIDE1{A}$$$$")


def test_the_mapped_atom_is_the_cap_of_an_unused_rgroup():
    """[NH2:2] leaves an amide when R2 is unused, not the acid it defaults to."""
    assert smiles("PEPTIDE1{[C[C@H](N[H:1])C([NH2:2])=O]}$$$$") == Chem.CanonSmiles(
        "C[C@H](N)C(N)=O"
    )


def test_a_mapped_leaving_atom_is_replaced_by_the_bond():
    molecule = Molecule(
        "PEPTIDE1{A}|CHEM1{[[H:1]OCCOCCO[H:2]]}$PEPTIDE1,CHEM1,1:R2-1:R1$$$"
    )

    assert CalcMolFormula(molecule.mol) == "C7H15NO4"
    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles("C[C@H](N)C(=O)OCCOCCO")


def test_a_mapped_rgroup_on_a_bracket_atom_keeps_its_hydrogens():
    """`[OH:2]` carries the hydroxyl hydrogen with it; nothing is left over."""
    mol = Molecule("PEPTIDE1{G.[C[C@H](N[H:1])C([OH:2])=O]}$$$$").mol

    assert all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms())
    assert all(atom.GetAtomicNum() != 1 for atom in mol.GetAtoms())
    assert CalcMolFormula(mol) == "C5H10N2O3"


@pytest.mark.parametrize(
    ("monomer", "message"),
    [
        ("[C[C@H](N[H:1])[C:2](O)=O]", "single bond"),
        ("[C[C@H](N[H:1])C(=[O:2])O]", "single bond"),
        ("[C[C@H](N[H:1])C([O-:2])=O]", "charged"),
        ("[C[C@H](N[H:1])C([OH:1])=O]", "more than one atom as R1"),
        ("[C[C@H](N[H:1])C([OH:5])=O]", "R1 to R4"),
        ("[C[C@H](N[*:1])C([*])=O |$;;;_R1;;_R2;$|]", "both an atom map and"),
    ],
)
def test_an_atom_map_that_cannot_be_an_rgroup_is_rejected(monomer, message):
    with pytest.raises(ValueError, match=message):
        Molecule(f"PEPTIDE1{{{monomer}}}$$$$")
