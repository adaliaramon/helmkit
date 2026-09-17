from helmkit import Molecule
from rdkit import Chem

# The structures the shipped monomer library builds for well known nucleosides.
# Written out here so the expectations do not depend on anything helmkit builds.

DEOXYADENOSINE = "C1[C@@H]([C@H](O[C@H]1N2C=NC3=C(N=CN=C32)N)CO)O"
ADENOSINE = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O"


def test_deoxyribose_builds_the_beta_D_sugar():
    """[dR] built the alpha anomer, with C4' left without a configuration.

    Every DNA strand is written with [dR], so every one of them came out as a
    stereoisomer of the strand the HELM string describes.
    """
    molecule = Molecule("RNA1{[dR](A)}$$$$")

    assert Chem.MolToInchi(molecule.mol) == Chem.MolToInchi(
        Chem.MolFromSmiles(DEOXYADENOSINE)
    )


def test_the_two_deoxyribose_symbols_agree():
    """The library holds 2'-deoxyribose twice, as [dR] and as [d].

    They are the same sugar, so they have to build the same nucleoside.
    """
    assert Chem.MolToSmiles(Molecule("RNA1{[dR](A)}$$$$").mol) == Chem.MolToSmiles(
        Molecule("RNA1{[d](A)}$$$$").mol
    )


def test_ribose_builds_adenosine():
    """A guard on the sugar the deoxy one is derived from."""
    molecule = Molecule("RNA1{R(A)}$$$$")

    assert Chem.MolToSmiles(molecule.mol) == Chem.CanonSmiles(ADENOSINE)
