import pytest
from helmkit import Molecule
from rdkit import Chem

# HELM2 writes a monomer repeated n times as A'n', and a repeated run of them
# as (A.G)'n'. Both used to be read as an unknown monomer name.


def smiles(helm):
    return Chem.MolToSmiles(Molecule(helm).mol)


@pytest.mark.parametrize(
    ("repeated", "written_out"),
    [
        ("PEPTIDE1{A'3'.G}$$$$", "PEPTIDE1{A.A.A.G}$$$$"),
        ("PEPTIDE1{G.A'1'}$$$$", "PEPTIDE1{G.A}$$$$"),
        ("PEPTIDE1{(A.G)'2'.C}$$$$", "PEPTIDE1{A.G.A.G.C}$$$$"),
        ("PEPTIDE1{(A.G).C}$$$$", "PEPTIDE1{A.G.C}$$$$"),
        ("PEPTIDE1{((A.G)'2'.C)'2'}$$$$", "PEPTIDE1{A.G.A.G.C.A.G.A.G.C}$$$$"),
        ("PEPTIDE1{[dK]'2'.A}$$$$", "PEPTIDE1{[dK].[dK].A}$$$$"),
        (
            "PEPTIDE1{[C[C@H](N[*])C([*])=O |$;;;_R1;;_R2;$|]'2'}$$$$",
            "PEPTIDE1{A.A}$$$$",
        ),
        (
            "RNA1{R(A)P.(R(G)P.R(U)P)'2'.R(C)}$$$$",
            "RNA1{R(A)P.R(G)P.R(U)P.R(G)P.R(U)P.R(C)}$$$$",
        ),
        ("RNA1{R(A)P'1'.R(C)}$$$$", "RNA1{R(A)P.R(C)}$$$$"),
        ("RNA1{R(A).P'1'}$$$$", "RNA1{R(A).P}$$$$"),
        # An annotation follows the repeat count.
        ("PEPTIDE1{A'2'\"x\".(G.C)'2'\"y\"}$$$$", "PEPTIDE1{A.A.G.C.G.C}$$$$"),
        ("PEPTIDE1{(A.C\"m\".G)'2'}$$$$", "PEPTIDE1{A.C.G.A.C.G}$$$$"),
    ],
)
def test_a_repeat_is_the_sequence_written_out(repeated, written_out):
    assert smiles(repeated) == smiles(written_out)


def test_connections_count_the_residues_a_repeat_writes_out():
    """TestValidation in the reference toolkit: residue 4 is the second C."""
    repeated = Molecule(
        "PEPTIDE1{F.L.C'3'}|PEPTIDE2{C.D}$PEPTIDE2,PEPTIDE1,1:R3-4:R3$$$"
    )
    written_out = Molecule(
        "PEPTIDE1{F.L.C.C.C}|PEPTIDE2{C.D}$PEPTIDE2,PEPTIDE1,1:R3-4:R3$$$"
    )

    assert Chem.MolToSmiles(repeated.mol) == Chem.MolToSmiles(written_out.mol)
    assert repeated.bondlist == written_out.bondlist
    assert len(repeated.residue_reps["PEPTIDE1"]) == 5


def test_a_connection_past_the_written_out_sequence_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        Molecule("PEPTIDE1{A'2'}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$")


@pytest.mark.parametrize(
    ("helm", "message"),
    [
        ("PEPTIDE1{A'3-5'}$$$$", "range"),
        ("PEPTIDE1{(A.G)'3-5'}$$$$", "range"),
        ("PEPTIDE1{A'0'}$$$$", "at least once"),
        ("PEPTIDE1{A'x'}$$$$", "not a number"),
        ("PEPTIDE1{A''}$$$$", "not a number"),
        ("PEPTIDE1{A'-1'}$$$$", "range"),
        ("PEPTIDE1{'3'}$$$$", "repeats nothing"),
        ("PEPTIDE1{A.'3'}$$$$", "repeats nothing"),
        ("PEPTIDE1{A'3}$$$$", "not in the aa monomer library"),
        ("PEPTIDE1{()'2'}$$$$", "no name"),
        # It is not settled whether this repeats the nucleotide or its phosphate.
        ("RNA1{R(A)P'2'}$$$$", "write it as a group"),
        ("CHEM1{[PEG2]'2'}$$$$", "exactly one residue"),
    ],
)
def test_a_repeat_that_is_not_one_molecule_is_rejected(helm, message):
    with pytest.raises(ValueError, match=message):
        Molecule(helm)


def test_alternatives_in_parentheses_are_not_a_group():
    """(A,G) and (A+G) list what a residue might be; they are not a sequence."""
    with pytest.raises(ValueError, match="not in the aa monomer library"):
        Molecule("PEPTIDE1{(A,G)'2'}$$$$")
    with pytest.raises(ValueError, match="not in the aa monomer library"):
        Molecule("PEPTIDE1{(A:1+G:2)}$$$$")
