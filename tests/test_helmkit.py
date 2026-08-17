from helmkit import Molecule


def test_extract_chain_id_rejects_trailing_characters():
    molecule = Molecule.__new__(Molecule)

    try:
        molecule._extract_polymer_type("PEPTIDE1foo")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid chain ID"
