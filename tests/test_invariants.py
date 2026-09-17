import random
import warnings

import pytest
from helmkit import Molecule
from rdkit import Chem
from rdkit import rdBase

# Generated HELM strings, close enough to valid to reach the parts of the parser
# that hand-written cases do not. Whatever goes in, helmkit has to either build
# a molecule that satisfies the invariants below or refuse with a ValueError.
# Anything else is a defect: a bare IndexError or an RDKit pre-condition
# violation tells a caller nothing, and a molecule that breaks an invariant is
# not the one the HELM string describes.

AMINO_ACIDS = ["A", "C", "D", "E", "F", "G", "K", "P", "R", "S", "W", "Y", "Aib"]
SUGARS = ["R", "dR"]
BASES = ["A", "C", "G", "U", "T"]
PHOSPHATES = ["P", "sP"]
INLINE = [
    "[*N[C@@H](C)C(=O)* |$_R1;;;;;;_R2$|]",
    "[*OCCO* |$_R1;;;;_R2$|]",
    "[*/C=C/C(=O)* |$_R1;;;;;_R2$|]",
]
MUTATIONS = '[](){}.|$,:R-0123456789ABC "*/'


def peptide_residue(rng):
    choice = rng.random()
    if choice < 0.55:
        return rng.choice(AMINO_ACIDS)
    if choice < 0.75:
        return f"[{rng.choice(AMINO_ACIDS)}]"
    if choice < 0.9:
        return rng.choice(INLINE)
    return f"({rng.choice(AMINO_ACIDS)},[{rng.choice(AMINO_ACIDS)}])"


def rna_residue(rng):
    residue = rng.choice(SUGARS)
    if rng.random() < 0.8:
        residue += f"({rng.choice(BASES)})"
    if rng.random() < 0.7:
        residue += rng.choice(PHOSPHATES)
    return residue


def chain(rng, number):
    kind = rng.choices(["PEPTIDE", "RNA", "CHEM"], [6, 3, 1])[0]
    if kind == "CHEM":
        return f"{kind}{number}{{{rng.choice(INLINE)}}}", 1
    pick = peptide_residue if kind == "PEPTIDE" else rna_residue
    residues = [pick(rng) for _ in range(rng.randint(1, 4))]
    return f"{kind}{number}{{{'.'.join(residues)}}}", len(residues)


def generate(rng):
    chains, sizes = [], []
    for number in range(1, rng.randint(1, 3) + 1):
        text, size = chain(rng, number)
        chains.append(text)
        sizes.append(size)

    names = [text[: text.index("{")] for text in chains]
    connections = [
        f"{names[a]},{names[b]},{rng.randint(1, sizes[a] + 1)}:R{rng.randint(1, 4)}"
        f"-{rng.randint(1, sizes[b] + 1)}:R{rng.randint(1, 4)}"
        for a, b in (
            (rng.randrange(len(chains)), rng.randrange(len(chains)))
            for _ in range(rng.randint(0, 2))
        )
    ]
    helm = "|".join(chains) + "$" + "|".join(connections) + "$$$"

    for _ in range(rng.choices([0, 1, 3], [7, 2, 1])[0]):
        if not helm:
            break
        at = rng.randrange(len(helm))
        roll = rng.random()
        if roll < 0.34:
            helm = helm[:at] + helm[at + 1 :]
        elif roll < 0.67:
            helm = helm[:at] + rng.choice(MUTATIONS) + helm[at:]
        else:
            helm = helm[:at] + rng.choice(MUTATIONS) + helm[at + 1 :]
    return helm


def check_invariants(molecule):
    """Everything that must hold for a molecule helmkit agreed to build."""
    atoms = molecule.mol.GetNumAtoms()

    assert len(molecule.offset) == len(molecule.monomers) + 1, "offset length"
    assert molecule.offset[-1] == atoms, "offsets do not cover the molecule"
    for i in range(len(molecule.monomers)):
        assert molecule.offset[i] < molecule.offset[i + 1], (
            f"monomer {i + 1} contributed no atoms"
        )

    indices = molecule.monomer_indices
    assert len(indices) == atoms, "one monomer index per atom"
    assert indices == sorted(indices), "monomer indices out of order"

    for (first, _, second, _), bond_index in zip(
        molecule.bondlist, molecule.bond_indices
    ):
        bond = molecule.mol.GetBondWithIdx(bond_index)
        joined = {indices[bond.GetBeginAtomIdx()], indices[bond.GetEndAtomIdx()]}
        assert joined == {first, second}, f"bond joins {joined}, not {{first, second}}"

    assert Chem.MolToSmiles(molecule.mol) is not None
    molecule.mol.GetRingInfo().NumRings()


@pytest.mark.parametrize("seed", range(4))
def test_generated_helm_either_builds_correctly_or_raises_value_error(seed):
    rng = random.Random(seed)

    for _ in range(400):
        helm = generate(rng)
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            try:
                with rdBase.BlockLogs():
                    molecule = Molecule(helm)
            except ValueError:
                continue
            except Exception as error:  # noqa: BLE001 - the point of the test
                pytest.fail(f"{helm!r} raised {type(error).__name__}: {error}")

            try:
                check_invariants(molecule)
            except AssertionError as error:
                pytest.fail(f"{helm!r}: {error}")

        assert not raised, (
            f"{helm!r} warned instead of failing: {[str(w.message) for w in raised]}"
        )
