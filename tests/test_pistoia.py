from pathlib import Path

import pytest
from helmkit import Molecule
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# Test cases from the Pistoia Alliance HELM2 reference implementation; see
# tests/data/pistoia/README.md for where each comes from.
DATA = Path(__file__).parent / "data" / "pistoia"


def load(name):
    """The HELM strings in a test case file, numbered by line from 1."""
    lines = (DATA / name).read_text(encoding="utf-8-sig").splitlines()
    # The first line is the heading "Positive" or "Negative".
    return [(n, line.strip()) for n, line in enumerate(lines[1:], 1) if line.strip()]


POSITIVE = load("positive_testcases.txt")
NEGATIVE = load("negative_testcases.txt")

# Positive cases that describe one concrete molecule, which helmkit builds.
# Every other one describes an abstract structure (a BLOB, an unknown monomer
# such as X, N or *, a ratio or an alternative, a polymer group, a connection to
# "any C"), a monomer that is not in the library, or chemistry the library
# cannot carry, and helmkit rejects it.
#
# Line 5 repeats a group of nucleotides, (R(G)P.R(U)P.R(G)P)'15', but ends in
# P(C), a base hung off a phosphate, which has no R3 to hang it from.
BUILT = {1, 2, 4, 25}


def sanitizable(mol):
    """Does the molecule survive RDKit sanitization unchanged in formula?"""
    copy = Chem.Mol(mol)
    Chem.SanitizeMol(copy)
    return CalcMolFormula(copy) == CalcMolFormula(mol)


# Syntax test cases


@pytest.mark.parametrize(
    ("line", "helm"), NEGATIVE, ids=[f"line{n}" for n, _ in NEGATIVE]
)
def test_notation_the_reference_parser_rejects_is_rejected(line, helm):
    with pytest.raises(ValueError):
        Molecule(helm)


@pytest.mark.parametrize(
    ("line", "helm"),
    [pytest.param(n, h, id=f"line{n}") for n, h in POSITIVE if n in BUILT],
)
def test_a_concrete_molecule_the_reference_parser_accepts_is_built(line, helm):
    assert sanitizable(Molecule(helm).mol)


@pytest.mark.parametrize(
    ("line", "helm"),
    [pytest.param(n, h, id=f"line{n}") for n, h in POSITIVE if n not in BUILT],
)
def test_an_abstract_structure_the_reference_parser_accepts_is_rejected(line, helm):
    """An abstract structure has no one molecule, so a ValueError saying why
    is the right answer; building something would be quietly wrong."""
    with pytest.raises(ValueError):
        Molecule(helm)


def test_a_repeated_group_of_nucleotides_from_line_5_is_built():
    """Line 5 without the P(C) helmkit's phosphate cannot carry."""
    helm = dict(POSITIVE)[5].replace(".P(C)}", "}")
    plain = helm.replace(
        "(R(G)P.R(U)P.R(G)P)'15'", ".".join(["R(G)P.R(U)P.R(G)P"] * 15)
    )

    assert Chem.MolToSmiles(Molecule(helm).mol) == Chem.MolToSmiles(Molecule(plain).mol)


# Chemistry test cases from HELM2NotationToolkit


def test_an_inline_monomer_written_with_bare_dummies_is_the_same_molecule():
    """SMILESTest.testGetSmilesFormats: `[*]` and `*` are the same R-group."""
    bracketed = Molecule(
        "PEPTIDE1{A.[[*]C(=O)[C@H](C)N([*])C |$_R2;;;;;;_R1;;;$|].A}$$$$V2.0"
    )
    bare = Molecule("PEPTIDE1{A.[*C(=O)[C@H](C)N(*)C |$_R2;;;;;;_R1;;;$|].A}$$$$V2.0")

    assert Chem.MolToSmiles(bare.mol) == Chem.MolToSmiles(bracketed.mol)
    assert CalcMolFormula(bare.mol) == "C10H19N3O4"


def test_an_inline_monomer_written_with_atom_maps_is_the_same_molecule():
    """SMILESTest.testGetSmilesFormats: `[H:1]` and `[OH:2]` mark the atoms an
    R-group replaces, and describe the same monomer as the CXSMILES labels."""
    labelled = Molecule(
        "PEPTIDE1{A.[[*]C(=O)[C@H](C)N([*])C |$_R2;;;;;;_R1;;;$|].A}$$$$V2.0"
    )
    mapped = Molecule("PEPTIDE1{A.[[OH:2]C(=O)[C@H](C)N([H:1])C].A}$$$$V2.0")

    assert Chem.MolToSmiles(mapped.mol) == Chem.MolToSmiles(labelled.mol)


def test_an_atom_mapped_alanine_is_alanine():
    """SMILESTest.testInlineNotation spells the first residue as SMILES."""
    library = Molecule("PEPTIDE1{A.G.G.G.C.C.K.K.K.K}$$$$")
    inline = Molecule("PEPTIDE1{[C[C@H](N[H:1])C([OH:2])=O].G.G.G.C.C.K.K.K.K}$$$$")

    assert Chem.MolToSmiles(inline.mol) == Chem.MolToSmiles(library.mol)


def test_an_inline_alanine_is_alanine():
    """SMILESTest.getInlineSmilesPeptideNotation against its library spelling."""
    library = Molecule("PEPTIDE1{G.G.K.A.A.[seC]}$$$$")
    inline = Molecule(
        "PEPTIDE1{G.G.K.A.[C[C@H](N[*])C([*])=O |$;;;_R1;;_R2;$|].[seC]}$$$$"
    )

    assert Chem.MolToSmiles(inline.mol) == Chem.MolToSmiles(library.mol)


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="a counter-ion is rejected as a monomer that falls into fragments",
)
@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{G.G.K.[[Na+].C[C@H](N[*])C([O-])[*] |$;;;;_R1;;;_R2$|].A.[seC]}$$$$",
        "RNA1{P.R(A)P.R(A)[[Na+].[O-]P([*])([*])=O |$;;;_R1;_R2;$|].R(C)}$$$$",
    ],
)
def test_an_inline_monomer_may_carry_a_counter_ion(helm):
    """SMILESTest builds inline salts; the ion was never joined by an R-group."""
    mol = Molecule(helm).mol

    assert "Na" in CalcMolFormula(mol)


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{A.A.G.K}$PEPTIDE1,PEPTIDE1,1:R1-4:R2$$$",
        "RNA1{R(C)P.RP.R(A)P.RP.R(A)P.R(U)P}$RNA1,RNA1,1:R1-16:R2$$$",
        "PEPTIDE1{A.A.C.G.[dK].E.C.H.A}$PEPTIDE1,PEPTIDE1,3:R3-7:R3$$$",
    ],
)
def test_a_cyclised_polymer_from_the_smiles_tests_is_built(helm):
    """SMILESTest.testHELM1AgainstHELM2 and testAttachmentMonomer."""
    mol = Molecule(helm).mol

    assert sanitizable(mol)
    assert mol.GetRingInfo().NumRings() > 0


DUPLEX_STRANDS = (
    "RNA1{R(A)P.R(U)P.R(C)P.R(C)P.R(A)P.R(A)P.R(A)P.R(G)P.R(A)P.R(U)P.R(A)P.R(C)P"
    ".R(U)P.R(A)P.R(G)P.R(C)P.R(U)P.R(U)P.R(U)P.R(G)P.R(C)P.R(A)P.R(G)P.R(A)P"
    ".R(A)P.R(U)P.R(G)}"
    "|RNA2{R(U)P.R(U)P.R(C)P.R(U)P.R(G)P.R(C)P.R(A)P.R(A)P.R(A)P.R(G)P.R(C)P"
    ".R(U)P.R(A)P.R(G)P.R(U)P.R(A)P.R(U)P.R(C)P.R(U)P.R(U)P.R(U)P.R(G)P.R(G)P"
    ".[dR](A)P.[dR](T)}"
)
DUPLEX_PAIRS = "|".join(
    f"RNA1,RNA2,{3 * i - 1}:pair-{77 - 3 * i}:pair" for i in range(1, 26)
)
# MoleculePropertyCalculatorTest.testgetMolecularFormularExamples
DUPLEX_FORMULA = "C495H611N191O359P50"


def test_the_strands_of_an_rna_duplex_have_the_reference_formula():
    mol = Molecule(f"{DUPLEX_STRANDS}$$$$").mol

    assert CalcMolFormula(mol) == DUPLEX_FORMULA


def test_hydrogen_bonds_in_their_own_section_leave_the_formula_alone():
    """HELM1 gives base pairs a section of their own."""
    molecule = Molecule(f"{DUPLEX_STRANDS}$${DUPLEX_PAIRS}$$")

    assert CalcMolFormula(molecule.mol) == DUPLEX_FORMULA
    assert len(molecule.hydrogen_bonds) == 25


def test_hydrogen_bonds_in_the_connection_section_leave_the_formula_alone():
    """MoleculePropertyCalculatorTest writes the duplex the HELM2 way."""
    molecule = Molecule(f"{DUPLEX_STRANDS}${DUPLEX_PAIRS}$$$V2.0")

    assert CalcMolFormula(molecule.mol) == DUPLEX_FORMULA
    assert len(molecule.hydrogen_bonds) == 25
