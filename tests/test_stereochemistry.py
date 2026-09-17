import pytest
from helmkit import Molecule
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Double bond geometry survived neither the way a monomer is parsed nor the
# deletion of the dummy atoms, and the corpus test cannot see it because it
# strips the /b layer of the InChI before comparing.

E_MONOMER = "[*/C=C/C(=O)* |$_R1;;;;;_R2$|]"
Z_MONOMER = "[*/C=C\\C(=O)* |$_R1;;;;;_R2$|]"
METHYL_MONOMER = "[*/C(C)=C/C(=O)* |$_R1;;;;;;_R2$|]"


def test_an_inline_monomer_keeps_the_geometry_it_declares():
    """Parsing without sanitizing never worked the geometry out at all."""
    molecule = Molecule(f"PEPTIDE1{{{E_MONOMER}.{E_MONOMER}}}$$$$")

    stereo = [
        bond.GetStereo()
        for bond in molecule.mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]
    assert stereo, "the double bond geometry was lost"


@pytest.mark.parametrize(
    ("monomer", "expected"),
    [(E_MONOMER, "C=CC(=O)/C=C/C(=O)O"), (Z_MONOMER, "C=CC(=O)/C=C\\C(=O)O")],
)
def test_a_backbone_bond_takes_over_the_stereo_reference(monomer, expected):
    """The bond stands where the R-group stood, so the geometry is unchanged."""
    molecule = Molecule(f"PEPTIDE1{{{monomer}.{monomer}}}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(expected)


def test_a_surviving_substituent_flips_the_stereo_reference():
    """It sits across the double bond from the R-group it replaces."""
    molecule = Molecule(f"PEPTIDE1{{{METHYL_MONOMER}}}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles("C/C=C\\C(=O)O")


def test_geometry_that_stops_being_defined_is_dropped():
    """Capping R1 with hydrogen leaves a CH2 end, which has no geometry."""
    molecule = Molecule(f"PEPTIDE1{{{E_MONOMER}}}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles("C=CC(=O)O")


def test_a_library_monomer_keeps_geometry_away_from_its_rgroups():
    molecule = Molecule("PEPTIDE1{A.[Me_Bmt(E)].G}$$$$")

    inchi = Chem.MolToInchi(molecule.mol)
    assert any(layer.startswith("b") for layer in inchi.split("/"))


def test_smiles_and_inchi_agree_about_the_geometry():
    """Stereo lives on the double bond; writing SMILES needs bond directions."""
    molecule = Molecule(f"PEPTIDE1{{{E_MONOMER}.{E_MONOMER}}}$$$$")

    smiles = Chem.MolToSmiles(molecule.mol)
    inchi = Chem.MolToInchi(molecule.mol)
    assert "/C=C/" in smiles or "/C=C\\" in smiles
    assert any(layer.startswith("b") for layer in inchi.split("/"))


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{A.R.G}$$$$",
        "PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$",
        "RNA1{R(A)P.R(C)}$$$$",
    ],
)
def test_ring_queries_work_on_the_finished_molecule(helm):
    """Ring membership was never perceived, so RDKit raised on any ring query."""
    mol = Molecule(helm).mol

    assert mol.GetRingInfo().NumRings() == rdMolDescriptors.CalcNumRings(mol)
    assert rdMolDescriptors.CalcTPSA(mol) > 0
    mol.GetSubstructMatches(Chem.MolFromSmarts("[R]"))
    mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1"))


def test_a_cyclic_peptide_reports_its_ring():
    linear = Molecule("PEPTIDE1{A.G.G}$$$$").mol
    cyclic = Molecule("PEPTIDE1{A.G.G}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$").mol

    assert linear.GetRingInfo().NumRings() == 0
    assert cyclic.GetRingInfo().NumRings() == 1
