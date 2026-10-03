import pytest
from rdkit import Chem

from helmkit import load_monomer_library
from helmkit import Molecule

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


@pytest.mark.parametrize(
    "helm",
    ["RNA1{R(A)P..R(C)}$$$$", "RNA1{R(A)P.}$$$$", "RNA1{.R(A)}$$$$", "RNA1{.}$$$$"],
)
def test_an_empty_rna_residue_is_rejected(helm):
    """An empty RNA residue used to drop out of the chain, or leave no molecule."""
    with pytest.raises(ValueError, match="has no name"):
        Molecule(helm)


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
    with pytest.raises(ValueError, match="bonded more than once"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R2-2:R1$$$")


def test_a_repeated_connection_is_rejected():
    bridge = "PEPTIDE1,PEPTIDE1,2:R3-4:R3"
    with pytest.raises(ValueError, match="bonded more than once"):
        Molecule(f"PEPTIDE1{{A.C.G.C}}${bridge}|{bridge}$$$")


def test_a_monomer_bonded_to_itself_is_rejected():
    with pytest.raises(ValueError, match="bonded more than once"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R1-1:R1$$$")


def test_two_bonds_landing_on_one_atom_are_rejected():
    """A phosphate carries R1 and R2 on the same atom, so two different
    R-groups can still put two bonds between the same pair of atoms."""
    with pytest.raises(ValueError, match="Duplicate bond"):
        Molecule("RNA1{P}|RNA2{P}$RNA1,RNA2,1:R1-1:R1|RNA1,RNA2,1:R2-1:R2$$$")


def test_a_bond_from_an_atom_back_to_itself_is_rejected():
    with pytest.raises(ValueError, match="bonded to itself"):
        Molecule("RNA1{P}$RNA1,RNA1,1:R1-1:R2$$$")


@pytest.mark.parametrize(
    "hydrogen_bond",
    [
        "PEPTIDE1,PEPTIDE1,x:pair-2:pair",
        "PEPTIDE1,PEPTIDE1,1:pair",
        "PEPTIDE1,PEPTIDE9,1:pair-2:pair",
        "PEPTIDE1,PEPTIDE1,1:pair-9:pair",
        "PEPTIDE1,PEPTIDE1,1:R2-2:R1",
        "PEPTIDE1,PEPTIDE1,1:pair-2:R1",
    ],
)
def test_a_malformed_hydrogen_bond_is_rejected(hydrogen_bond):
    with pytest.raises(ValueError):
        Molecule(f"PEPTIDE1{{A.G}}$${hydrogen_bond}$$")


@pytest.mark.parametrize(
    "hydrogen_bond",
    [
        "PEPTIDE1,PEPTIDE1,x:pair-2:pair",
        "PEPTIDE1,PEPTIDE9,1:pair-2:pair",
        "PEPTIDE1,PEPTIDE1,1:pair-9:pair",
        "PEPTIDE1,PEPTIDE1,1:pair-2:R1",
        "PEPTIDE1,PEPTIDE1,1:R2-2:pair",
    ],
)
def test_a_malformed_hydrogen_bond_among_the_connections_is_rejected(hydrogen_bond):
    """HELM2 writes hydrogen bonds in the connection section."""
    with pytest.raises(ValueError):
        Molecule(f"PEPTIDE1{{A.G}}${hydrogen_bond}$$$")


def test_a_hydrogen_bond_among_the_connections_is_recorded_not_bonded():
    paired = Molecule("RNA1{R(A)}|RNA2{R(U)}$RNA1,RNA2,2:pair-2:pair$$$V2.0")
    unpaired = Molecule("RNA1{R(A)}|RNA2{R(U)}$$$$V2.0")

    assert paired.hydrogen_bonds == [["RNA1", 1, "RNA2", 1]]
    assert paired.bondlist == unpaired.bondlist
    assert Chem.MolToSmiles(paired.mol) == Chem.MolToSmiles(unpaired.mol)


def test_hydrogen_bonds_and_covalent_bonds_mix_in_the_connection_section():
    molecule = Molecule(
        "RNA1{R(A)P}|RNA2{R(U)P}$RNA1,RNA2,2:pair-2:pair|RNA1,RNA2,3:R2-1:R1$$$V2.0"
    )

    assert molecule.hydrogen_bonds == [["RNA1", 1, "RNA2", 1]]
    assert len(molecule.bond_indices) == 5


def test_a_valid_hydrogen_bond_is_still_recorded():
    molecule = Molecule("RNA1{R(A)}|RNA2{R(U)}$$RNA1,RNA2,1:pair-1:pair$$")

    assert molecule.hydrogen_bonds == [["RNA1", 0, "RNA2", 0]]


def test_a_branch_monomer_cannot_start_a_chain():
    """`RNA1{(A)R}` used to build two disconnected fragments."""
    with pytest.raises(ValueError, match="nothing to attach to"):
        Molecule("RNA1{(A)R}$$$$")


@pytest.mark.parametrize(
    ("monomer", "atom"),
    [("*NCC(=O)CC* |$_R1;;;;;_R2;$|", "C"), ("*NCC(=O)* |$_R1;;;;_R2;$|", "O")],
)
def test_an_rgroup_label_on_a_real_atom_is_rejected(monomer, atom):
    """A label one place out used to make that atom the R-group.

    The atom stayed in the molecule as well as the bond made in its place, and
    the dummy atom the label was meant for was deleted.
    """
    with pytest.raises(ValueError, match=f"labels a {atom} atom _R2"):
        Molecule(f"PEPTIDE1{{A.[{monomer}].G}}$$$$")


def test_a_disconnected_inline_monomer_is_rejected():
    with pytest.raises(ValueError, match="falls into separate fragments"):
        Molecule("PEPTIDE1{[CCO.CCO]}$$$$")


def test_an_rgroup_that_bridges_two_halves_of_a_monomer_is_rejected():
    """The two halves only come apart once the R-group is deleted."""
    with pytest.raises(ValueError, match="falls into separate fragments"):
        Molecule(r"CHEM1{[CC*CC |$;;_R1;;$|]}$$$$")


def test_inline_monomers_do_not_grow_the_shared_library():
    """Inline SMILES monomers used to be written into the cached library."""
    library = load_monomer_library()
    before = len(library["aa"])

    Molecule("PEPTIDE1{[*N[C@@H](CCCCCCO)C(=O)* |$_R1;;;;;;;;;;;;_R2$|]}$$$$")

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
    "monomer", ["[** |$_R1;_R2$|]", "[*** |$_R1;;_R2$|]", "[* |$_R1$|]", "[*]"]
)
def test_a_monomer_of_nothing_but_rgroups_is_rejected(monomer):
    """It contributed no atoms at all and dropped out of the molecule, while
    the bonds recorded for it pointed at atoms that were no longer there."""
    with pytest.raises(ValueError, match="no atoms besides its R-groups"):
        Molecule(f"PEPTIDE1{{A.{monomer}.G}}$$$$")


def test_an_rgroup_bonded_only_to_dummies_is_rejected():
    """Its attachment point would be deleted along with the other dummy atoms.

    The monomer here keeps a real atom, so it is the attachment point rather
    than the whole monomer that would disappear.
    """
    with pytest.raises(ValueError, match="no attachment point"):
        Molecule(r"PEPTIDE1{A.[**C |$_R1;_R2;$|].G}$$$$")


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
        "CHEM1{[*OCCO* |$_R1;;;;;_R2$|]}|PEPTIDE1{A.C}$PEPTIDE1,CHEM1,2:R3-1:R1$$$"
    )

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@@H](CSOCCO)C(=O)O"
    )


def test_a_polymer_of_a_single_dummy_monomer_is_rejected():
    """The one monomer was dropped and an empty molecule came back instead."""
    with pytest.raises(ValueError, match="no atoms besides its R-groups"):
        Molecule("PEPTIDE1{[*]}$$$$")


def test_a_dummy_atom_without_an_rgroup_label_is_rejected():
    """Every dummy atom is deleted, so an unlabelled one vanished: here the R2
    cap was never applied and the residue came back as an aldehyde."""
    with pytest.raises(ValueError, match=r"dummy atom .* no _R<number> label"):
        Molecule("PEPTIDE1{[*N[C@@H](C)C(=O)* |$_R1;;;;;;$|]}$$$$")


@pytest.mark.parametrize(
    "group",
    [
        "G1(PEPTIDE1+PEPTIDE2)",
        "G1(PEPTIDE1+PEPTIDE2:2.5)",
        "G1(PEPTIDE1,PEPTIDE2)",
        "G1(PEPTIDE1:40+PEPTIDE2:60)",
        "G1(PEPTIDE1+PEPTIDE2)|G2(G1+PEPTIDE1:4.5)",
        'G1(PEPTIDE1+PEPTIDE2)"a note"',
    ],
)
def test_a_polymer_group_is_rejected_as_not_one_molecule(group):
    """HELM2 gives the third section to polymer groups, which describe a
    mixture or a choice of polymers. They used to be read as malformed
    hydrogen bonds, which said nothing about what was wrong."""
    with pytest.raises(ValueError, match="mixture or a choice of polymers"):
        Molecule(f"PEPTIDE1{{A}}|PEPTIDE2{{G}}$${group}$$V2.0")


@pytest.mark.parametrize("helm", ["CHEM1{[Dig]}$$$$", "PEPTIDE1{A.[Xyz].G}$$$$"])
def test_an_unknown_bracketed_symbol_is_rejected_quietly(helm, capfd):
    """It is tried as SMILES, and RDKit used to print its parse errors too."""
    with pytest.raises(ValueError, match="not in monomer library and is not a valid"):
        Molecule(helm)

    assert "SMILES Parse Error" not in capfd.readouterr().err
