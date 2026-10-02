import pytest
from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# A salt may carry its counter-ion as a fragment of its own. It used to be
# rejected as a monomer that falls apart; it is only a second piece of the
# monomer when it is neutral or carries an R-group of its own.

SODIUM_PHOSPHATE = "[[Na+].[O-]P([*])([*])=O |$;;;_R1;_R2;$|]"


def test_a_sodium_phosphate_is_built_with_its_counter_ion():
    salt = Molecule(f"RNA1{{R(A){SODIUM_PHOSPHATE}.R(C)}}$$$$")
    acid = Molecule("RNA1{R(A)P.R(C)}$$$$")

    assert CalcMolFormula(salt.mol) == "C19H24N8NaO11P"
    assert CalcMolFormula(acid.mol) == "C19H25N8O11P"
    assert len(Chem.GetMolFrags(salt.mol)) == 2
    assert len(salt.bond_indices) == len(acid.bond_indices)


def test_the_counter_ion_belongs_to_its_monomer():
    molecule = Molecule(f"RNA1{{R(A){SODIUM_PHOSPHATE}.R(C)}}$$$$")

    sodium = next(a.GetIdx() for a in molecule.mol.GetAtoms() if a.GetSymbol() == "Na")
    owner = molecule.monomers[molecule.monomer_indices[sodium]]
    assert owner["m_abbr"] == SODIUM_PHOSPHATE[1:-1]


def test_an_amine_salt_keeps_its_chloride():
    molecule = Molecule("PEPTIDE1{A.[[Cl-].[*]CC[NH3+] |$;_R1;;;$|]}$$$$")

    assert CalcMolFormula(molecule.mol) == "C5H13ClN2O"


@pytest.mark.parametrize(
    "monomer",
    [
        # Two neutral pieces, neither of them an ion.
        "[CCO.CCO]",
        # A neutral piece beside the monomer is not a counter-ion.
        "[O.[*]CC[NH3+] |$;_R1;;;$|]",
        # Each charged piece carries an R-group, so each is part of the monomer.
        "[[*]CC[NH3+].[*]CC(=O)[O-] |$_R1;;;;_R2;;;;$|]",
    ],
)
def test_a_second_piece_that_is_not_a_counter_ion_is_rejected(monomer):
    with pytest.raises(ValueError, match="falls into separate fragments"):
        Molecule(f"CHEM1{{{monomer}}}$$$$")


def test_a_library_monomer_may_carry_a_counter_ion(tmp_path):
    path = tmp_path / "salt.sdf"
    with Chem.SDWriter(str(path)) as writer:
        mol = Chem.MolFromSmiles("[Na+].*NCC(=O)[O-]")
        mol.SetProp("symbol", "GNa")
        mol.SetProp("m_type", "aa")
        mol.SetProp("m_RgroupIdx", "1,None,None,None")
        mol.SetProp("m_Rgroups", "H,None,None,None")
        writer.write(mol)

    molecule = Molecule("PEPTIDE1{GNa}$$$$", load_monomer_library(str(path)))

    assert CalcMolFormula(molecule.mol) == "C2H4NNaO2"
