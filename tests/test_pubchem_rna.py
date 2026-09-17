import collections
import gzip
import json
import re
from pathlib import Path

from helmkit import Molecule
from rdkit import Chem
from rdkit import rdBase

# Nucleic acid structures from PubChem, one record per distinct HELM string
# with every structure PubChem records for it. PubChem frequently holds several
# stereoisomers under one HELM string, so a structure counts as correct when it
# matches any of them.
DATA = Path(__file__).parent / "data" / "rna.ndjson.gz"

# Where the corpus stood when this test was written. A fall in matches, or a
# rise in mismatches, means something that used to be built correctly is not
# any more.
EXPECTED_MATCHES = 3032
ALLOWED_MISMATCHES = 195


def load():
    with gzip.open(DATA, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def layers(inchi):
    return {part[0]: part for part in (inchi or "").split("/")[1:] if part}


def without_protonation(inchi):
    """Drop the /p layer.

    It records the protonation state of the structure PubChem happens to hold,
    which a HELM string does not describe.
    """
    return re.sub(r"/p[+-]?\d*", "", inchi or "")


def states_the_stereochemistry(reference, built):
    """Can this reference settle the stereochemistry of what we built?

    Not when it is isotope labelled, not when it leaves a centre undefined, and
    not when one of the two has a tetrahedral layer and the other does not.
    """
    reference_layers, built_layers = layers(reference), layers(built)
    if "i" in reference_layers:
        return False
    if ("t" in built_layers) != ("t" in reference_layers):
        return False
    return "?" not in (reference_layers.get("t") or "")


def test_nucleic_acids_from_pubchem():
    """Build every HELM string PubChem records a nucleic acid for.

    Each one either matches a reference structure or falls into a category
    that says why the reference cannot settle it. A HELM string that fails to
    build for any reason other than a monomer the library does not carry, or
    that raises anything other than ValueError, fails the test outright. The remaining mismatches are
    HELM strings PubChem also maps to a stereoisomer the string itself does not
    determine; about eighty of them are the 2'-fluoro sugar FR, where the
    monomer library and PubChem disagree about which face the fluorine is on.
    """
    counts = collections.Counter()
    mismatched = []

    for record in load():
        helm = record["helm"]
        references = [r["inchi"] for r in record["refs"]]
        try:
            with rdBase.BlockLogs():
                built = Chem.MolToInchi(Molecule(helm).mol)
        except ValueError as error:
            assert "monomer library" in str(error), f"{helm}: {error}"
            counts["monomer not in the library"] += 1
            continue

        if any(
            without_protonation(built) == without_protonation(r) for r in references
        ):
            counts["match"] += 1
            continue

        settled = {
            without_protonation(r)
            for r in references
            if states_the_stereochemistry(r, built)
        }
        if not settled:
            counts["no reference states the stereochemistry"] += 1
        elif len(settled) > 1:
            counts["references disagree with each other"] += 1
        else:
            counts["mismatch"] += 1
            mismatched.append((helm, built, next(iter(settled))))

    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    assert counts["match"] >= EXPECTED_MATCHES, (
        f"fewer structures match than before ({counts['match']} < {EXPECTED_MATCHES}). {summary}"
    )
    assert counts["mismatch"] <= ALLOWED_MISMATCHES, (
        f"more structures mismatch than before ({counts['mismatch']} > {ALLOWED_MISMATCHES}). "
        f"{summary}\nfirst: {mismatched[:2]}"
    )
