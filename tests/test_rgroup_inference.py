import pytest
from helmkit import Molecule
from helmkit.molecule import _create_missing_monomer
from rdkit import Chem

# A monomer written as inline SMILES without _R1 and _R2 labels has its
# attachment points inferred. They are the amine and the carboxyl on the alpha
# carbon, so both are looked for there. Searching the whole molecule for an
# amine or a carbonyl instead finds a side chain one just as readily, and picks
# it whenever it happens to be the only one of its kind.

# Standard residues written out as plain SMILES, beside the library symbol for
# the same residue. Writing a residue either way has to give the same molecule.
SAME_AS_LIBRARY = [
    ("A", "N[C@@H](C)C(=O)O"),
    ("G", "NCC(=O)O"),
    ("P", "OC(=O)[C@@H]1CCCN1"),
    ("W", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("Q", "N[C@@H](CCC(N)=O)C(=O)O"),
    ("K", "N[C@@H](CCCCN)C(=O)O"),
    ("S", "N[C@@H](CO)C(=O)O"),
    ("F", "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("Orn", "N[C@@H](CCCN)C(=O)O"),
]


def tripeptide(monomer):
    return Chem.MolToSmiles(Molecule(f"PEPTIDE1{{A.[{monomer}].G}}$$$$").mol)


@pytest.mark.parametrize(
    ("symbol", "smiles"), SAME_AS_LIBRARY, ids=[s for s, _ in SAME_AS_LIBRARY]
)
def test_an_inline_residue_matches_the_library_symbol(symbol, smiles):
    """Glutamine, lysine and ornithine used to fail outright.

    Each carries a second nitrogen on its side chain, which left the amine
    ambiguous and no R1 to bond through.
    """
    assert tripeptide(smiles) == tripeptide(symbol)


def test_the_backbone_amine_is_preferred_to_a_side_chain_amine():
    """N-methyl-lysine has a secondary amine on its side chain.

    A secondary amine used to be looked for first, so the peptide bond was made
    to the side chain and the backbone amine was left free, without a word.
    """
    assert tripeptide("N[C@@H](CCCCNC)C(=O)O") == Chem.CanonSmiles(
        "CNCCCC[C@H](NC(=O)[C@H](C)N)C(=O)NCC(=O)O"
    )


def test_the_backbone_carboxyl_is_preferred_to_a_side_chain_aldehyde():
    """An aldehyde used to be looked for before a carboxylic acid.

    Aspartate semialdehyde has one on its side chain, so the next residue was
    bonded there and the backbone carboxyl was left free.
    """
    assert tripeptide("N[C@@H](CC=O)C(=O)O") == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@@H](CC=O)C(=O)NCC(=O)O"
    )


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # a backbone amine that is already substituted, as in sarcosine
        ("CNCC(=O)O", "C[C@H](N)C(=O)N(C)CC(=O)NCC(=O)O"),
        # a C terminus written as an aldehyde rather than an acid
        ("NCC=O", "C[C@H](N)C(=O)NCC(=O)NCC(=O)O"),
        # a side chain ketone, which is not a candidate carboxyl
        ("N[C@@H](CC(C)=O)C(=O)O", "CC(=O)C[C@H](NC(=O)[C@H](C)N)C(=O)NCC(=O)O"),
    ],
)
def test_shapes_that_already_worked_still_do(smiles, expected):
    assert tripeptide(smiles) == Chem.CanonSmiles(expected)


def test_a_monomer_with_no_usable_amine_is_still_refused():
    """Inference must not invent an attachment point that is not there."""
    with pytest.raises(ValueError, match="R-group 1 is not present"):
        Molecule("PEPTIDE1{A.[CCCC(=O)O].G}$$$$")


def labelled(core, first="_R1", last="_R2"):
    """CXSMILES giving the first and last atom of `core` an R-group label.

    The label block needs one field per atom, so the separators are counted
    rather than written out.
    """
    atoms = Chem.MolFromSmiles(core, sanitize=False).GetNumAtoms()
    return f"{core} |${first}{';' * (atoms - 1)}{last}$|"


def test_both_stray_hydroxyls_are_dropped():
    """A carboxyl written with its hydroxyl still on it beside an R-group is
    over-full, so the hydroxyl is dropped to make room.

    They used to be removed lowest index first. Removing an atom shifts every
    index above it, so with two of them the second removal took whatever had
    moved into that slot, which was the R-group dummy rather than the hydroxyl.
    """
    monomer = _create_missing_monomer(labelled("*C(=O)(O)CCCC(=O)(O)*"))

    assert sum(1 for i in monomer["m_RgroupIdx"] if i is not None) == 2, (
        "an R-group was deleted along with the hydroxyls"
    )
    assert Chem.MolToSmiles(monomer["m_romol"]) == Chem.MolToSmiles(
        Chem.MolFromSmiles("*C(=O)CCCC(*)=O")
    )
