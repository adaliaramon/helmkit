import pytest
from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit import Chem


def glycine_library(path, m_rgroups):
    """Write a one-monomer SDF library for glycine with the given cap groups."""
    with Chem.SDWriter(str(path)) as writer:
        mol = Chem.MolFromSmiles("*NCC(=O)*")
        mol.SetProp("symbol", "G")
        mol.SetProp("m_abbr", "Gly")
        mol.SetProp("m_type", "aa")
        mol.SetProp("m_RgroupIdx", "0,5,None,None")
        mol.SetProp("m_Rgroups", m_rgroups)
        writer.write(mol)

    return load_monomer_library(str(path))


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_amine_cap_group_is_added(tmp_path):
    """An NH2 cap becomes a nitrogen instead of disappearing with the R-group."""
    library = glycine_library(tmp_path / "amide.sdf", "H,NH2,None,None")

    molecule = Molecule("PEPTIDE1{G}$$$$", library)

    assert Chem.MolToSmiles(molecule.mol) == canonical("NCC(N)=O")


def test_oxygen_cap_group_written_without_its_hydrogen(tmp_path):
    """`O` and `OH` describe the same cap: the hydrogens follow from valence."""
    library = glycine_library(tmp_path / "acid.sdf", "H,O,None,None")

    molecule = Molecule("PEPTIDE1{G}$$$$", library)

    assert Chem.MolToSmiles(molecule.mol) == canonical("NCC(O)=O")


def test_cap_group_of_more_than_one_heavy_atom_is_rejected(tmp_path):
    """A cap that cannot be applied would silently cost the molecule an atom."""
    library = glycine_library(tmp_path / "ester.sdf", "H,OCC,None,None")

    with pytest.raises(ValueError, match="Unsupported R-group cap group OCC"):
        Molecule("PEPTIDE1{G}$$$$", library)


def test_terminal_phosphoramidate_keeps_its_amine():
    """A 5'-terminal m2np caps R1 with NH2, which used to be dropped silently."""
    molecule = Molecule("RNA1{[m2np].R(A)}$$$$")

    phosphorus = next(a for a in molecule.mol.GetAtoms() if a.GetSymbol() == "P")
    assert phosphorus.GetTotalNumHs() == 0
    assert sorted(n.GetSymbol() for n in phosphorus.GetNeighbors()) == [
        "N",
        "N",
        "O",
        "O",
    ]


def test_hydroxyl_caps_still_close_the_peptide_terminus():
    molecule = Molecule("PEPTIDE1{A.R.G}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == canonical(
        "C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O"
    )


def test_an_inline_amino_acid_agrees_with_the_same_monomer_from_the_library():
    """An unused R2 is the carboxyl terminus, so it caps with OH, not hydrogen.

    Without the cap the dummy was simply deleted and the carboxyl carbon kept
    only its double bonded oxygen, turning the residue into an aldehyde.
    """
    alanine = "[*N[C@@H](C)C(=O)* |$_R1;;;;;;_R2$|]"

    from_library = Molecule("PEPTIDE1{A}$$$$")
    from_smiles = Molecule(f"PEPTIDE1{{{alanine}}}$$$$")

    assert Chem.MolToSmiles(from_smiles.mol) == Chem.MolToSmiles(from_library.mol)
    assert Chem.MolToSmiles(from_smiles.mol) == canonical("C[C@H](N)C(=O)O")


def test_an_inline_amino_acid_at_the_end_of_a_chain_keeps_its_acid():
    alanine = "[*N[C@@H](C)C(=O)* |$_R1;;;;;;_R2$|]"

    molecule = Molecule(f"PEPTIDE1{{A.{alanine}}}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == canonical("C[C@H](N)C(=O)N[C@@H](C)C(=O)O")


def test_a_monomer_written_as_a_free_acid_does_not_gain_a_second_hydroxyl():
    """R2 inferred onto an existing carboxyl must not also be capped with OH."""
    molecule = Molecule("PEPTIDE1{A.[NCC(=O)O]}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == canonical("C[C@H](N)C(=O)NCC(=O)O")
