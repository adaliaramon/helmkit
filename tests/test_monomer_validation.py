import pytest
from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit import Chem

# A monomer library is read straight from an SDF, so its R-group properties can
# disagree with the molecules they describe. Every case here used to escape as
# an IndexError, a TypeError, an OverflowError or an RDKit range error, or to
# build a molecule that was quietly not the monomer the library described.

GLYCINE = "*NCC(=O)*"  # atoms: 0=* 1=N 2=C 3=C 4=O 5=*


def write_library(path, **overrides):
    """Write a one-monomer SDF library, overriding any of its properties."""
    props = {
        "symbol": "G",
        "m_abbr": "Gly",
        "m_type": "aa",
        "m_RgroupIdx": "0,5,None,None",
        "m_Rgroups": "H,OH,None,None",
    }
    props.update(overrides)
    with Chem.SDWriter(str(path)) as writer:
        mol = Chem.MolFromSmiles(GLYCINE)
        for key, value in props.items():
            mol.SetProp(key, value)
        writer.write(mol)
    return str(path)


def build(path):
    return Molecule("PEPTIDE1{G.G}$$$$", load_monomer_library(path))


def test_a_well_formed_library_still_works(tmp_path):
    molecule = build(write_library(tmp_path / "ok.sdf"))

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles("NCC(=O)NCC(=O)O")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"m_RgroupIdx": "0,99,None,None"}, "outside the 6 atoms"),
        ({"m_RgroupIdx": "0,-1,None,None"}, "outside the 6 atoms"),
        ({"m_RgroupIdx": "0,999999999999,None,None"}, "outside the 6 atoms"),
        ({"m_RgroupIdx": "0,x,None,None"}, "not a whole number"),
        ({"m_RgroupIdx": "0,5.0,None,None"}, "not a whole number"),
        ({"m_RgroupIdx": "0,None,None,None"}, "no atom index"),
        ({"m_Rgroups": "H,OH,OH,OH"}, "no atom index"),
        ({"m_Rgroups": "H"}, "1 R-group cap groups but 4"),
        ({"m_Rgroups": ""}, "0 R-group cap groups but 4"),
        ({"m_RgroupIdx": ""}, "4 R-group cap groups but 0"),
        ({"m_RgroupIdx": "0,5,2,3", "m_Rgroups": "H,OH"}, "2 R-group cap groups but 4"),
    ],
)
def test_a_library_that_disagrees_with_itself_is_rejected(tmp_path, overrides, message):
    path = write_library(tmp_path / "bad.sdf", **overrides)

    with pytest.raises(ValueError, match=message):
        build(path)


def test_an_rgroup_that_is_not_a_dummy_atom_is_rejected(tmp_path):
    """Pointing R1 at the nitrogen used to graft a spare nitrogen into the chain."""
    path = write_library(tmp_path / "real-atom.sdf", m_RgroupIdx="1,5,None,None")

    with pytest.raises(ValueError, match="is not a dummy atom"):
        build(path)


def test_two_rgroups_on_one_atom_are_rejected(tmp_path):
    """R1 and R2 sharing an atom used to build an unrelated molecule."""
    path = write_library(tmp_path / "shared.sdf", m_RgroupIdx="0,0,None,None")

    with pytest.raises(ValueError, match="already R-group 1"):
        build(path)


def test_a_cap_group_beyond_the_fourth_rgroup_is_applied(tmp_path):
    """Capping used to stop after R4, silently dropping any later cap group."""
    path = tmp_path / "five.sdf"
    with Chem.SDWriter(str(path)) as writer:
        # atoms: 0=* 1=C 2=* 3=* 4=C 5=* 6=* 7=N
        mol = Chem.MolFromSmiles("*C(*)(*)C(*)(*)N")
        mol.SetProp("symbol", "X")
        mol.SetProp("m_abbr", "Xaa")
        mol.SetProp("m_type", "aa")
        mol.SetProp("m_RgroupIdx", "0,2,3,5,6")
        mol.SetProp("m_Rgroups", "H,H,H,H,OH")
        writer.write(mol)

    molecule = Molecule("PEPTIDE1{X}$$$$", load_monomer_library(str(path)))

    oxygens = sum(a.GetSymbol() == "O" for a in molecule.mol.GetAtoms())
    assert oxygens == 1, "the R5 cap group was dropped"


def test_a_cap_group_beyond_the_fourth_rgroup_is_still_validated(tmp_path):
    path = write_library(
        tmp_path / "five-bad.sdf",
        m_RgroupIdx="0,5,None,None,None",
        m_Rgroups="H,OH,None,None,OH",
    )

    with pytest.raises(ValueError, match=r"R-group 5 .* no atom index"):
        build(path)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ("_R1;;;;;_R1", "more than one atom _R1"),
        ("_R;;;;;_R2", "not of the form _R<number>"),
        ("_R0;;;;;_R2", "R-groups run from R1 to R4"),
        ("_R9;;;;;_R2", "R-groups run from R1 to R4"),
    ],
)
def test_bad_inline_rgroup_labels_are_rejected(labels, message):
    """_R0 and _R9 used to be dropped silently; the others died inside RDKit."""
    monomer = f"[{GLYCINE} |${labels}$|]"

    with pytest.raises(ValueError, match=message):
        Molecule(f"PEPTIDE1{{{monomer}}}$$$$")


def test_a_well_formed_inline_monomer_still_works():
    monomer = f"[{GLYCINE} |$_R1;;;;;_R2$|]"

    molecule = Molecule(f"PEPTIDE1{{{monomer}}}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles("NCC(=O)O")


def test_a_monomer_with_too_few_rgroups_is_rejected_in_a_peptide(tmp_path):
    """The peptide backbone used to index the attachment points unguarded."""
    path = tmp_path / "one-rgroup.sdf"
    with Chem.SDWriter(str(path)) as writer:
        mol = Chem.MolFromSmiles("*NCC=O")  # atoms: 0=* 1=N 2=C 3=C 4=O
        mol.SetProp("symbol", "G")
        mol.SetProp("m_abbr", "Gly")
        mol.SetProp("m_type", "aa")
        mol.SetProp("m_RgroupIdx", "0")
        mol.SetProp("m_Rgroups", "H")
        writer.write(mol)

    with pytest.raises(ValueError, match="R-group 2 is not present in monomer 1"):
        Molecule("PEPTIDE1{G.G}$$$$", load_monomer_library(str(path)))


def test_a_monomer_with_no_rgroups_is_rejected_in_a_peptide(tmp_path):
    path = tmp_path / "no-rgroups.sdf"
    with Chem.SDWriter(str(path)) as writer:
        mol = Chem.MolFromSmiles("NCC=O")
        mol.SetProp("symbol", "G")
        mol.SetProp("m_abbr", "Gly")
        mol.SetProp("m_type", "aa")
        mol.SetProp("m_RgroupIdx", "")
        mol.SetProp("m_Rgroups", "")
        writer.write(mol)

    with pytest.raises(ValueError, match="R-group 2 is not present"):
        Molecule("PEPTIDE1{G.G}$$$$", load_monomer_library(str(path)))
