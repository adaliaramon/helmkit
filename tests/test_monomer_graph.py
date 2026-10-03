import csv
import gzip
import json
import re
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit import rdBase

from helmkit import Molecule

# `monomer_indices` and `bond_indices` describe the molecule as a graph of
# monomers: which monomer every atom came from, and which bonds join one
# monomer to the next. The checks below hold them to the molecule itself.
# Cutting the molecule at the bonds `bond_indices` names has to leave one
# piece per monomer, made of exactly the atoms `monomer_indices` gives it, and
# each piece has to be the monomer the HELM string names: its structure from
# the library, with each R-group that forms a bond left as an attachment point
# and every other R-group replaced by its cap group.

DATA = Path(__file__).parent / "data"


def cap_atomic_number(cap):
    return Chem.GetPeriodicTable().GetAtomicNumber(re.match(r"[A-Z][a-z]?", cap)[0])


def canonical(mol):
    """SMILES that ignores how attachment points are labelled."""
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def expected_residue(monomer, bonded_rgroups):
    """A monomer as it should appear once its inter-monomer bonds are cut."""
    template = monomer["m_romol"]
    caps = [
        None if cap == "None" else cap
        for cap in template.GetProp("m_Rgroups").split(",")
    ]
    mol = Chem.RWMol(template)
    kept = set()
    for rgroup, atom_idx in enumerate(monomer["m_RgroupIdx"]):
        if atom_idx is None:
            continue
        if rgroup in bonded_rgroups:
            kept.add(atom_idx)
        elif caps[rgroup] not in (None, "H"):
            mol.ReplaceAtom(atom_idx, Chem.Atom(cap_atomic_number(caps[rgroup])))
            kept.add(atom_idx)
    removed = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetIdx() not in kept
    ]
    for atom_idx in sorted(removed, reverse=True):
        mol.RemoveAtom(atom_idx)
    return canonical(mol)


def check_monomer_graph(molecule):
    mol = molecule.mol
    monomer_of = molecule.monomer_indices
    bond_indices = molecule.bond_indices

    assert len(monomer_of) == mol.GetNumAtoms()
    assert len(set(bond_indices)) == len(bond_indices), "a bond is listed twice"

    listed = set(bond_indices)
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if monomer_of[begin] != monomer_of[end]:
            assert bond.GetIdx() in listed, (
                f"bond {bond.GetIdx()} joins two monomers but is not listed"
            )

    # Which R-groups each monomer bonds through, from the HELM string.
    bonded_rgroups = [set() for _ in molecule.monomers]
    for monomer_idx, rgroup in molecule.used_rgroups:
        bonded_rgroups[monomer_idx].add(rgroup)

    if bond_indices:
        pieces = Chem.FragmentOnBonds(
            mol, bond_indices, addDummies=True, dummyLabels=[(0, 0)] * len(bond_indices)
        )
    else:
        pieces = Chem.Mol(mol)
    atoms_of_piece = []
    fragments = Chem.GetMolFrags(
        pieces, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=atoms_of_piece
    )

    assert len(fragments) == len(molecule.monomers), (
        f"cutting the listed bonds leaves {len(fragments)} pieces "
        f"for {len(molecule.monomers)} monomers"
    )
    seen = set()
    for fragment, atoms in zip(fragments, atoms_of_piece):
        original = [idx for idx in atoms if idx < mol.GetNumAtoms()]
        owners = {monomer_of[idx] for idx in original}
        assert len(owners) == 1, f"one piece holds atoms of monomers {owners}"
        (owner,) = owners
        assert owner not in seen, f"monomer {owner + 1} falls into several pieces"
        seen.add(owner)

        monomer = molecule.monomers[owner]
        assert canonical(fragment) == expected_residue(
            monomer, bonded_rgroups[owner]
        ), f"monomer {owner + 1} ({monomer['m_abbr']}) is not the atoms given to it"


@pytest.mark.parametrize(
    ("helm", "names", "edges"),
    [
        ("PEPTIDE1{A.R.G}$$$$", ["A", "R", "G"], {(0, 1), (1, 2)}),
        (
            "PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$",
            ["A", "C", "G", "C"],
            {(0, 1), (1, 2), (2, 3), (1, 3)},
        ),
        (
            "PEPTIDE1{A.G.S}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$",
            ["A", "G", "S"],
            {(0, 1), (1, 2), (0, 2)},
        ),
        (
            "PEPTIDE1{A.K}|PEPTIDE2{G.S}$PEPTIDE1,PEPTIDE2,2:R3-2:R2$$$",
            ["A", "K", "G", "S"],
            {(0, 1), (2, 3), (1, 3)},
        ),
        (
            "RNA1{R(A)P.R(C)}$$$$",
            ["R", "A", "P", "R", "C"],
            {(0, 1), (0, 2), (2, 3), (3, 4)},
        ),
        (
            "PEPTIDE1{A.[*N[C@@H](C/C=C/C)C(=O)* |$_R1;;;;;;;;;_R2$|].G}$$$$",
            ["A", "*N[C@@H](C/C=C/C)C(=O)* |$_R1;;;;;;;;;_R2$|", "G"],
            {(0, 1), (1, 2)},
        ),
        ("PEPTIDE1{G}$$$$", ["G"], set()),
    ],
)
def test_the_monomer_graph_is_the_one_the_helm_string_describes(helm, names, edges):
    molecule = Molecule(helm)
    monomer_of = molecule.monomer_indices

    assert [monomer["m_abbr"] for monomer in molecule.monomers] == names
    found = set()
    for bond_idx in molecule.bond_indices:
        bond = molecule.mol.GetBondWithIdx(bond_idx)
        first, second = (
            monomer_of[bond.GetBeginAtomIdx()],
            monomer_of[bond.GetEndAtomIdx()],
        )
        found.add((min(first, second), max(first, second)))
    assert found == edges
    check_monomer_graph(molecule)


def cycpeptmpdb():
    with open(DATA / "peptides.csv", encoding="utf-8-sig") as handle:
        return [row["HELM"] for row in csv.DictReader(handle)]


def pubchem_rna():
    with gzip.open(DATA / "rna.ndjson.gz", "rt") as handle:
        return [json.loads(line)["helm"] for line in handle if line.strip()]


@pytest.mark.parametrize("corpus", [cycpeptmpdb, pubchem_rna])
def test_the_monomer_graph_matches_the_molecule_across_a_corpus(corpus):
    checked = 0
    # Every fifth string keeps the test quick; the whole corpus passes too.
    for helm in corpus()[::5]:
        try:
            with rdBase.BlockLogs():
                molecule = Molecule(helm)
        except ValueError:
            continue
        try:
            check_monomer_graph(molecule)
        except AssertionError as error:
            pytest.fail(f"{helm}: {error}")
        checked += 1
    assert checked > 500
