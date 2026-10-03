import pytest
from rdkit import Chem

from helmkit import load_monomer_library
from helmkit import Molecule


def write_library(path, monomers):
    """Write a minimal SDF monomer library with the given extra properties."""
    with Chem.SDWriter(str(path)) as writer:
        for props in monomers:
            mol = Chem.MolFromSmiles("*NCC(=O)*")
            mol.SetProp("m_RgroupIdx", "0,5,None,None")
            mol.SetProp("m_Rgroups", "H,OH,None,None")
            for key, value in props.items():
                mol.SetProp(key, value)
            writer.write(mol)


def test_monomers_without_abbreviation_are_kept(tmp_path):
    """m_abbr is optional, so a library omitting it must still load."""
    library_path = tmp_path / "no-abbr.sdf"
    write_library(library_path, [{"symbol": "G", "m_type": "aa"}])

    library = load_monomer_library(str(library_path))

    assert list(library["aa"]) == ["G"]
    assert library["aa"]["G"]["m_abbr"] == "G"


def test_abbreviation_is_used_when_present(tmp_path):
    library_path = tmp_path / "with-abbr.sdf"
    write_library(library_path, [{"symbol": "G", "m_type": "aa", "m_abbr": "Gly"}])

    library = load_monomer_library(str(library_path))

    assert library["aa"]["G"]["m_abbr"] == "Gly"


def test_monomer_without_symbol_is_skipped_with_a_warning(tmp_path):
    library_path = tmp_path / "no-symbol.sdf"
    write_library(library_path, [{"m_type": "aa"}, {"symbol": "G", "m_type": "aa"}])

    with pytest.warns(UserWarning, match="symbol"):
        library = load_monomer_library(str(library_path))

    assert list(library["aa"]) == ["G"]


def test_building_a_molecule_leaves_the_library_unchanged():
    """Monomers are shared with the library, so bonding must not modify them."""
    library = load_monomer_library()
    before = {
        m_type: {symbol: list(m["m_Rgroups"]) for symbol, m in monomers.items()}
        for m_type, monomers in library.items()
    }

    Molecule("PEPTIDE1{A.G.C}|PEPTIDE2{A}$PEPTIDE1,PEPTIDE2,3:R3-1:R1$$$", library)

    after = {
        m_type: {symbol: list(m["m_Rgroups"]) for symbol, m in monomers.items()}
        for m_type, monomers in library.items()
    }
    assert after == before
