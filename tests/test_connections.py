import pytest
from helmkit import Molecule
from rdkit import Chem


def test_residue_number_zero_is_rejected():
    """HELM residues are numbered from 1, so 0 must not wrap to the last one."""
    with pytest.raises(ValueError, match=r"Residue number 0 .* is not positive"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,0:R1-2:R2$$$")


def test_residue_number_past_the_end_of_the_chain_is_rejected():
    with pytest.raises(ValueError, match="Residue 5 is out of range"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R1-5:R2$$$")


def test_connection_to_an_undeclared_chain_is_rejected():
    with pytest.raises(ValueError, match="PEPTIDE9"):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE9,1:R1-1:R1$$$")


def test_rgroup_number_zero_is_rejected():
    """R0 must not index the attachment points from the end of the list."""
    with pytest.raises(ValueError, match=r"R-group R0 .* is not positive"):
        Molecule("RNA1{R(A)P.R(C)P}$RNA1,RNA1,1:R0-4:R3$$$")


def test_missing_rgroup_is_reported_with_the_number_used_in_the_helm():
    with pytest.raises(
        ValueError, match=r"R-group 4 is not present in monomer 1 \(A\)"
    ):
        Molecule("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R4-2:R2$$$")


def test_valid_connections_are_unaffected():
    """The head-to-tail cycle and the disulfide bridge must still be built."""
    cycle = Molecule("PEPTIDE1{A.G.G}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$")
    assert Chem.MolToSmiles(cycle.mol) == Chem.CanonSmiles(
        "C[C@@H]1NC(=O)CNC(=O)CNC1=O"
    )

    bridged = Molecule("PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$")
    assert Chem.MolToSmiles(bridged.mol) == Chem.CanonSmiles(
        "C[C@H](N)C(=O)N[C@H]1CSSC[C@@H](C(=O)O)NC(=O)CNC1=O"
    )
    assert len(bridged.bondlist) == 4


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{A.C.A}$PEPTIDE1,PEPTIDE1,1:R2-3:R1$$$",
        "PEPTIDE1{F.W.A.E}$PEPTIDE1,PEPTIDE1,3:R1-3:R2|PEPTIDE1,PEPTIDE1,3:R2-4:R3$$$",
        "RNA1{[dR](C)P}$RNA1,RNA1,1:R3-1:R2$$$",
        "RNA1{R(A)(C)}$$$$",
    ],
)
def test_an_rgroup_cannot_be_bonded_twice(helm):
    """An R-group stands for one attachment, so a second bond must be refused.

    Both bonds landed on the same atom and left it with five bonds, so the
    molecule could not be sanitized and could not be read back from its own
    SMILES. The last case needs no connection section at all: it is two bases
    written onto one sugar.
    """
    with pytest.raises(ValueError, match="bonded more than once"):
        Molecule(helm)
