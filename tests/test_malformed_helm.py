import pytest
from helmkit import load_monomer_library
from helmkit import Molecule
from rdkit import Chem

# Malformed HELM must raise, never produce a quietly different molecule. The
# cases below all used to warn and carry on, or to escape as an exception that
# says nothing about which part of the input is at fault.


@pytest.mark.parametrize(
    "helm", ["", "$$$$", "hello world", "PEPTIDE1", "PEPTIDE1{A.G", "PEPTIDE1A.G}"]
)
def test_a_string_without_a_polymer_is_rejected(helm):
    """These used to warn and hand back an empty molecule."""
    with pytest.raises(ValueError, match="is not of the form"):
        Molecule(helm)


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="empty sequence"):
        Molecule("PEPTIDE1{}$$$$")


@pytest.mark.parametrize("helm", ["PEPTIDE1{A}|$$$$", "PEPTIDE1{A}||PEPTIDE2{G}$$$$"])
def test_an_empty_chain_slot_is_rejected(helm):
    """A stray separator used to drop silently and leave the rest standing."""
    with pytest.raises(ValueError, match="is not of the form"):
        Molecule(helm)


def test_duplicate_chain_id_is_still_caught_after_an_empty_chain():
    with pytest.raises(ValueError):
        Molecule("PEPTIDE1{}|PEPTIDE1{G}$$$$")


def test_text_after_the_sequence_is_rejected():
    """`PEPTIDE1{A.G}junk` used to build A.G and ignore the rest."""
    with pytest.raises(ValueError, match="Unexpected text junk"):
        Molecule("PEPTIDE1{A.G}junk$$$$")


def test_a_polymer_annotation_is_still_accepted():
    """HELM2 allows a quoted annotation after the sequence."""
    molecule = Molecule('PEPTIDE1{A.G}"a note"$$$$')

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles("C[C@H](N)C(=O)NCC(=O)O")


def test_trailing_separator_is_rejected():
    """`A.G.` used to drop the empty residue; `.A.G` already complained."""
    with pytest.raises(ValueError, match="Monomer 3 has no name"):
        Molecule("PEPTIDE1{A.G.}$$$$")


@pytest.mark.parametrize("sequence", ["A]G", "A[G", "R(AP", "RA)P"])
def test_unbalanced_brackets_are_rejected(sequence):
    with pytest.raises(ValueError, match="Unbalanced brackets"):
        Molecule(f"PEPTIDE1{{{sequence}}}$$$$")


@pytest.mark.parametrize(
    "connection",
    [
        "PEPTIDE1,PEPTIDE1",
        "PEPTIDE1,PEPTIDE1,1:R1-2:R2,x",
        "PEPTIDE1,PEPTIDE1,garbage",
        "PEPTIDE1,PEPTIDE1,x:R1-2:R2",
        "PEPTIDE1,PEPTIDE1,?:R1-2:R2",
        "PEPTIDE1,PEPTIDE1,-1:R1-2:R2",
        "PEPTIDE1,PEPTIDE1,1:X-2:R2",
        "PEPTIDE1,PEPTIDE1,1:R1R-2:R2",
        ",,",
    ],
)
def test_an_unparseable_connection_is_rejected(connection):
    """Skipping the connection returned the linear peptide with no bridge."""
    with pytest.raises(ValueError):
        Molecule(f"PEPTIDE1{{A.C.G.C}}${connection}$$$")


def test_a_connection_that_repeats_the_backbone_bond_is_rejected():
    """RDKit answered this with a bare C++ pre-condition violation."""
    with pytest.raises(ValueError, match="Duplicate bond"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R2-2:R1$$$")


def test_a_repeated_connection_is_rejected():
    bridge = "PEPTIDE1,PEPTIDE1,2:R3-4:R3"
    with pytest.raises(ValueError, match="Duplicate bond"):
        Molecule(f"PEPTIDE1{{A.C.G.C}}${bridge}|{bridge}$$$")


def test_a_monomer_bonded_to_itself_is_rejected():
    with pytest.raises(ValueError, match="bonded to itself"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R1-1:R1$$$")


@pytest.mark.parametrize(
    "hydrogen_bond",
    [
        "PEPTIDE1,PEPTIDE1,x:pair-2:pair",
        "PEPTIDE1,PEPTIDE1,1:pair",
        "PEPTIDE1,PEPTIDE9,1:pair-2:pair",
        "PEPTIDE1,PEPTIDE1,1:pair-9:pair",
    ],
)
def test_a_malformed_hydrogen_bond_is_rejected(hydrogen_bond):
    with pytest.raises(ValueError):
        Molecule(f"PEPTIDE1{{A.G}}$${hydrogen_bond}$$")


def test_a_valid_hydrogen_bond_is_still_recorded():
    molecule = Molecule("RNA1{R(A)}|RNA2{R(U)}$$RNA1,RNA2,1:pair-1:pair$$")

    assert molecule.hydrogen_bonds == [["RNA1", 0, "RNA2", 0]]


def test_a_branch_monomer_cannot_start_a_chain():
    """`RNA1{(A)R}` used to build two disconnected fragments."""
    with pytest.raises(ValueError, match="nothing to attach to"):
        Molecule("RNA1{(A)R}$$$$")


def test_a_disconnected_inline_monomer_is_rejected():
    with pytest.raises(ValueError, match="not a single connected fragment"):
        Molecule("PEPTIDE1{[CCO.CCO]}$$$$")


def test_inline_monomers_do_not_grow_the_shared_library():
    """Inline SMILES monomers used to be written into the cached library."""
    library = load_monomer_library()
    before = len(library["aa"])

    Molecule("PEPTIDE1{[*N[C@@H](CCCCCCO)C(=O)* |$_R1;;;;;;;;;;;_R2$|]}$$$$")

    assert len(library["aa"]) == before


@pytest.mark.parametrize("symbol", ["B", "J", "O", "U", "X", "Z"])
def test_a_bare_unknown_symbol_is_not_read_as_smiles(symbol):
    """B, J, O, U, X and Z are not amino acids, so they are likely typos.

    Every one of them is also a valid SMILES string, and `PEPTIDE1{B}` used to
    build a lone boron atom rather than complain.
    """
    with pytest.raises(ValueError, match="not in the aa monomer library"):
        Molecule(f"PEPTIDE1{{{symbol}}}$$$$")


def test_a_bare_smiles_string_is_rejected():
    with pytest.raises(ValueError, match="square brackets"):
        Molecule("PEPTIDE1{NCC(=O)O}$$$$")


def test_a_bracketed_smiles_string_is_still_accepted():
    """Inline SMILES is how a HELM string names a monomer outside the library."""
    molecule = Molecule("PEPTIDE1{A.[*N[C@@H](CO)C(=O)* |$_R1;;;;;;;_R2$|].G}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@@H](CO)C(=O)NCC(=O)O"
    )


def test_an_unknown_bracketed_symbol_still_reports_the_symbol():
    with pytest.raises(ValueError, match="Monomer Zzz not in monomer library"):
        Molecule("PEPTIDE1{[Zzz]}$$$$")


def test_ambiguous_monomers_are_still_resolved():
    """`(S,[dS])` picks the bracketed alternative and flags the molecule."""
    molecule = Molecule("PEPTIDE1{A.(S,[dS]).G}$$$$")

    assert molecule.has_ambiguous_monomers
    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@H](CO)C(=O)NCC(=O)O"
    )


@pytest.mark.parametrize(
    "monomer", ["[** |$_R1;_R2$|]", "[*** |$_R1;;_R2$|]", "[* |$_R1$|]"]
)
def test_an_rgroup_bonded_only_to_dummies_is_rejected(monomer):
    """Its attachment point would be deleted along with the other dummy atoms.

    The monomer then contributed no atoms at all and dropped out of the
    molecule, while the bonds recorded for it pointed at atoms that were gone.
    """
    with pytest.raises(ValueError, match="no attachment point"):
        Molecule(f"PEPTIDE1{{A.{monomer}.G}}$$$$")


def test_a_monomer_never_vanishes_from_the_molecule():
    """Every monomer has to contribute at least one atom to the result."""
    molecule = Molecule("PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$")

    assert molecule.offset[-1] == molecule.mol.GetNumAtoms()
    for i in range(len(molecule.monomers)):
        assert molecule.offset[i] < molecule.offset[i + 1]
    # every recorded bond is really in the molecule
    assert len(molecule.bond_indices) == len(molecule.bondlist)


@pytest.mark.parametrize("tail", ["", "V2.0", "]", "V2.0]", "a]b", '{"a":[1]}'])
def test_a_later_section_containing_a_bracket_does_not_break_the_split(tail):
    """A closing bracket in a trailing section used to suppress the $ split."""
    bridged = "PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$"

    molecule = Molecule(f"{bridged}${tail}")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@H]1CSSC[C@@H](C(=O)O)NC(=O)CNC1=O"
    )


def test_separators_inside_an_inline_monomer_are_not_section_separators():
    """CXSMILES carries both `$` and `|` inside the brackets of a monomer."""
    molecule = Molecule(
        "CHEM1{[*OCCO* |$_R1;;;;_R2$|]}|PEPTIDE1{A.C}$PEPTIDE1,CHEM1,2:R3-1:R1$$$"
    )

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@@H](CSOCCO)C(=O)O"
    )
