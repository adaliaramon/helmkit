import pytest
from helmkit import Molecule
from rdkit import Chem

# An RNA chain writes several monomers into one residue, so it is split by its
# own routine rather than the one peptides use. The two have to agree about
# what a monomer name looks like when they hand it on, including its brackets:
# _process_monomer tells an inline SMILES monomer from a library symbol by
# whether it is bracketed, so a splitter that strips them makes every inline
# monomer in an RNA chain look like an unknown symbol.

INLINE = "[*OCC* |$_R1;;;_R2$|]"
TRIPHOSPHATE = "[*P(=O)(O)OP(=O)(O)OP(=O)(O)O |$_R2;;;;;;;;;;;;|]"


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("R(A)", ["R", "(A)"]),
        ("[dR](A)P", ["[dR]", "(A)", "P"]),
        ("R([5meC])P", ["R", "([5meC])", "P"]),
        (f"{INLINE}R(A)", [INLINE, "R", "(A)"]),
    ],
)
def test_splitting_a_residue_keeps_the_brackets_on(sequence, expected):
    """The brackets are what say the name is a SMILES string, not a symbol."""
    assert Molecule._parse_rna_string(sequence) == expected


def test_an_inline_monomer_is_accepted_in_either_kind_of_chain():
    """Whether a monomer may be written as SMILES cannot depend on the chain.

    The peptide splitter leaves brackets alone and the RNA one used to strip
    them, so the same monomer was accepted in one chain and refused in the
    other.
    """
    Molecule(f"PEPTIDE1{{{INLINE}}}$$$$")
    Molecule(f"RNA1{{{INLINE}}}$$$$")


def test_an_inline_monomer_builds_in_an_rna_chain():
    """A 5' triphosphate is written as an inline SMILES monomer."""
    molecule = Molecule(f"RNA1{{{TRIPHOSPHATE}R(A)}}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"
    )


@pytest.mark.parametrize("helm", ["RNA1{B}$$$$", "RNA1{Zz}$$$$", "RNA1{NCC(=O)O}$$$$"])
def test_a_bare_name_in_an_rna_chain_is_still_a_library_lookup(helm):
    """Keeping the brackets must not let an unbracketed name mean SMILES."""
    with pytest.raises(ValueError, match="not in the rna monomer library"):
        Molecule(helm)
