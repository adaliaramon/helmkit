import gzip
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit import rdBase

from helmkit import load_monomer_library
from helmkit import Molecule

# The HELM core monomer library of the Pistoia Alliance, which the HELM project
# recommends as everyone's base. Its SMILES mark each R-group with an atom map
# on the atom that leaves, so helmkit can build each one as an inline monomer
# and compare it with its own monomer of the same symbol, caps and all.
DATA = Path(__file__).parent / "data" / "pistoia" / "HELMCoreLibrary.ndjson.gz"

POLYMERS = {"PEPTIDE": ("aa", "PEPTIDE1"), "RNA": ("rna", "RNA1")}

# Monomers that come out differently, and why. Each one has been looked at; a
# new difference fails the test, and so does one of these that goes away.
KNOWN_DIFFERENCES = {
    # The core library's SMILES describes something its name does not.
    "m2np": "Phosphorodiamidate, but the SMILES has a carbon for the phosphorus",
    "nC62r": "the SMILES puts a peroxide on the 3' oxygen",
    "m3U": "the SMILES has an extra methyl on C5, which is m3T",
    "e5U": "the SMILES has a saturated ring",
    "e6A": "the SMILES is O6-ethyl-2-aminopurine, not N6-ethyladenine",
    # Only the cap of an unused R-group differs: H here, OH in helmkit, as
    # for every other backbone linker.
    "me": "R1 and R2 capped with H",
    "sp": "R1 and R2 capped with H",
    # The name states the configuration, and helmkit agrees with it.
    "Ss2cEt": "(S)-cEt is S in helmkit and R in the core library",
    "cet": "(S)-cEt is S in helmkit and R in the core library",
    "Scmoe": "(S)-cMOE is S in helmkit and R in the core library",
    "Smclna": "(2S)-methyl-cLNA is S in helmkit and R in the core library",
    "Sm5lna": "(5S)-5-methyl-LNA is S in helmkit and R in the core library",
    "Sm5ALlna": "(5S)-5-methyl-alpha-L-LNA is S in helmkit, R in the core library",
    "Sm5moe": "(5S)-methyl-2'-O-MOE is S in helmkit and R in the core library",
    "Rm5fl2r": "(R)-2-fluoro-5-methylribose is R in helmkit, S in the core library",
    "Rm5d": "(R)-5-methyldeoxyribose is R in helmkit, undefined in the core library",
    "Sm5d": "(S)-5-methyldeoxyribose is S in helmkit, undefined in the core library",
    "Rmn2cet": "amino-(R)-cEt is R in helmkit and S in the core library",
    "Llyspna": "L-lysine is S in helmkit and R in the core library",
    # Stereochemistry the name does not settle.
    "afhna": "the fluorine of 3-ara-FHNA is on opposite faces",
    "5R6Sm5cEt": "the two methyl-bearing centres disagree",
    "5S6Rm5cEt": "the two methyl-bearing centres disagree",
    "5S6Sm5cEt": "the two methyl-bearing centres disagree",
    "Liprglyol2r": "the glycerol centre disagrees",
    "She5d": "the hydroxyethyl centre disagrees",
    "SRpabCNA": "the phosphorus and C5 centres disagree",
    "r": "C4 of the sugar disagrees",
    "Smc": "the base-bearing centre disagrees",
    "trina2": "two ring centres disagree",
    # helmkit's library leaves the carbon that carries the base undefined.
    "Nmc": "the base-bearing carbon is undefined in helmkit",
    "fl2Nmc": "the base-bearing carbon is undefined in helmkit",
    "afl2Nmc": "the base-bearing carbon is undefined in helmkit",
}


def shared_monomers():
    library = load_monomer_library()
    with gzip.open(DATA, "rt", encoding="utf-8") as handle:
        core = [json.loads(line) for line in handle if line.strip()]
    return [
        pytest.param(m, id=f"{m['polymerType']}-{m['symbol']}")
        for m in core
        if m["polymerType"] in POLYMERS
        and m["symbol"] in library.get(POLYMERS[m["polymerType"]][0], {})
    ]


def inchi(helm):
    with rdBase.BlockLogs():
        return Chem.MolToInchi(Molecule(helm).mol)


@pytest.mark.parametrize("monomer", shared_monomers())
def test_a_monomer_agrees_with_the_core_library(monomer):
    polymer = POLYMERS[monomer["polymerType"]][1]
    ours = inchi(f"{polymer}{{[{monomer['symbol']}]}}$$$$")
    theirs = inchi(f"{polymer}{{[{monomer['smiles']}]}}$$$$")

    if monomer["symbol"] in KNOWN_DIFFERENCES:
        assert ours != theirs, (
            f"no longer differs: {KNOWN_DIFFERENCES[monomer['symbol']]}"
        )
    else:
        assert ours == theirs


def test_the_comparison_covers_the_shared_monomers():
    """A change in the library or the data should not quietly empty the test."""
    assert len(shared_monomers()) == 511
