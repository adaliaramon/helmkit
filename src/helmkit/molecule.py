import bisect
import multiprocessing
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files
from typing import assert_never
from typing import cast
from typing import Literal
from typing import overload
from typing import TypedDict
from typing import TypeVar

from rdkit import Chem
from rdkit import rdBase


MAX_RGROUPS = 4

_FLIPPED_STEREO = {
    Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS,
}


def get_molecule_property(
    molecule: Chem.Mol, property_name: str, default: str | None = None
) -> str | None:
    return molecule.GetProp(property_name, default=default)


T = TypeVar("T")


@overload
def parse_comma_separated_property(
    molecule: Chem.Mol, property_name: str, convert_func: None = None
) -> list[str | None]: ...


@overload
def parse_comma_separated_property(
    molecule: Chem.Mol, property_name: str, convert_func: Callable[[str], T]
) -> list[T | None]: ...


def parse_comma_separated_property(
    molecule: Chem.Mol,
    property_name: str,
    convert_func: Callable[[str], T] | None = None,
) -> list[str | None] | list[T | None]:
    property_value = get_molecule_property(molecule, property_name)
    if not property_value:
        return []

    values = property_value.split(",")
    if convert_func:
        return [convert_func(v) if v != "None" else None for v in values]
    return [None if v == "None" else v for v in values]


def infer_attachment_points(
    molecule: Chem.Mol, rgroup_indices: Sequence[int | None], name: str = "monomer"
) -> list[int | None]:
    """Infer attachment points by finding atoms bonded to R-group atoms."""
    attachment_points: list[int | None] = []

    for rgroup, r_idx in enumerate(rgroup_indices, start=1):
        if r_idx is None:
            attachment_points.append(None)
            continue

        atom = molecule.GetAtomWithIdx(r_idx)
        bonds: tuple[Chem.Bond, ...] = atom.GetBonds()

        # The attachment point has to be a real atom. Dummy atoms are all
        # deleted while sanitizing, so an R-group bonded only to other dummies
        # would leave the bonds made to it pointing at atoms that are gone and
        # drop the monomer out of the molecule without a word.
        for bond in bonds:
            other_idx = bond.GetOtherAtomIdx(r_idx)
            if molecule.GetAtomWithIdx(other_idx).GetAtomicNum() != 0:
                attachment_points.append(other_idx)
                break
        else:
            raise ValueError(
                f"R-group {rgroup} of {name} (atom {r_idx}) is not bonded to a non-dummy atom, so it has no attachment point."
            )

    return attachment_points


def validate_rgroups(
    symbol: str,
    molecule: Chem.Mol,
    rgroups: Sequence[str | None],
    rgroup_idx: Sequence[int | None],
) -> None:
    """Check that a monomer's R-group properties describe its structure.

    Without this the indices are used directly to look atoms up and to pair
    caps with atoms, so a library that disagrees with its own molecules either
    fails deep inside RDKit or builds a molecule that is quietly not the
    monomer the library meant to describe.
    """
    if len(rgroups) != len(rgroup_idx):
        raise ValueError(
            f"Monomer {symbol} lists {len(rgroups)} R-group cap groups but {len(rgroup_idx)} R-group atom indices."
        )

    num_atoms = molecule.GetNumAtoms()
    seen: dict[int, int] = {}

    for rgroup, (cap, idx) in enumerate(zip(rgroups, rgroup_idx), start=1):
        if idx is None:
            if cap is not None:
                raise ValueError(
                    f"R-group {rgroup} of monomer {symbol} has the cap group {cap} but no atom index."
                )
            continue
        if not 0 <= idx < num_atoms:
            raise ValueError(
                f"R-group {rgroup} of monomer {symbol} is atom {idx}, which is outside the {num_atoms} atoms of the monomer."
            )
        if molecule.GetAtomWithIdx(idx).GetAtomicNum() != 0:
            raise ValueError(
                f"R-group {rgroup} of monomer {symbol} is atom {idx}, which is not a dummy atom."
            )
        if idx in seen:
            raise ValueError(
                f"R-group {rgroup} of monomer {symbol} is atom {idx}, which is already R-group {seen[idx]}."
            )
        seen[idx] = rgroup


def validate_monomer_core(name: str, molecule: Chem.Mol) -> None:
    """Check what the monomer is left as once its R-groups are gone.

    Every dummy atom is deleted while sanitizing, so a monomer made only of
    R-groups disappears out of the molecule, and one whose R-group sits between
    two halves falls into two pieces. Neither says anything at the time.
    """
    core = Chem.DeleteSubstructs(Chem.Mol(molecule), Chem.MolFromSmarts("[#0]"))
    if core.GetNumAtoms() == 0:
        raise ValueError(f"Monomer {name} has no atoms besides its R-groups.")
    if len(Chem.GetMolFrags(core)) > 1:
        raise ValueError(
            f"Monomer {name} falls into separate fragments once its R-groups are removed."
        )


class MonomerData(TypedDict):
    m_romol: Chem.Mol
    m_Rgroups: list[str | None]
    m_RgroupIdx: list[int | None]
    m_attachmentPointIdx: list[int | None]
    m_type: str
    m_abbr: str


MonomerLibrary = dict[str, dict[str, MonomerData]]


@lru_cache
def load_monomer_library(library_path: str | None = None) -> MonomerLibrary:
    """Load and prepare monomer data from SDF file."""
    if library_path is None:
        library_path = str(files("helmkit.data") / "monomers.sdf")
    monomers_dict: MonomerLibrary = defaultdict(dict)
    supplier = Chem.SDMolSupplier(library_path, removeHs=False)

    for mol in cast(Iterable[Chem.Mol | None], supplier):
        if mol is None:
            continue

        symbol = get_molecule_property(mol, "symbol")
        if not symbol:
            warnings.warn("Monomer without a symbol property will be skipped")
            continue

        m_type = get_molecule_property(mol, "m_type", "")
        if m_type not in ["aa", "rna", "chem"]:
            warnings.warn(
                f"Monomer {symbol} has unknown type {m_type} and will be skipped"
            )
            continue

        rgroups = parse_comma_separated_property(mol, "m_Rgroups")
        try:
            rgroup_idx = parse_comma_separated_property(mol, "m_RgroupIdx", int)
        except ValueError as e:
            raise ValueError(
                f"Monomer {symbol} has an m_RgroupIdx that is not a whole number: {e}"
            ) from None
        validate_rgroups(symbol, mol, rgroups, rgroup_idx)
        validate_monomer_core(symbol, mol)
        attachment_point_idx = infer_attachment_points(mol, rgroup_idx, symbol)

        # m_abbr is only used for display, so fall back to the symbol when a
        # library does not provide it instead of dropping the monomer.
        abbr = get_molecule_property(mol, "m_abbr") or symbol

        monomers_dict[m_type][symbol] = {
            "m_romol": mol,
            "m_Rgroups": rgroups,
            "m_RgroupIdx": rgroup_idx,
            "m_attachmentPointIdx": attachment_point_idx,
            "m_type": m_type,
            "m_abbr": abbr,
        }

    return monomers_dict


# Bounded: the cache key is an arbitrary monomer name, so an unbounded cache
# would keep an entry for every distinct inline SMILES string ever parsed.
@lru_cache(maxsize=4096)
def _is_free_carbonyl_carbon(molecule: Chem.Mol, idx: int) -> bool:
    """Is this a carbonyl carbon that does not already carry a second oxygen?"""
    atom = molecule.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return False

    carbonyl = False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() != 8:
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            carbonyl = True
        else:
            return False
    return carbonyl


def _create_missing_monomer(monomer_name: str, m_type: str = "aa") -> MonomerData:
    mol = Chem.MolFromSmiles(monomer_name, sanitize=False)
    if mol is None:
        if monomer_name.endswith("|") and not monomer_name.endswith("$|"):
            return _create_missing_monomer(monomer_name[:-1] + "$|", m_type)
        raise ValueError(
            f"Monomer {monomer_name} not in monomer library and is not a valid SMILES string"
        )
    with rdBase.BlockLogs():
        error = Chem.SanitizeMol(mol, catchErrors=True)
    if error == Chem.SanitizeFlags.SANITIZE_PROPERTIES:
        mol = Chem.RWMol(mol)
        pattern = Chem.MolFromSmarts("O[CX4]=O")
        matches = mol.GetSubstructMatches(pattern)
        for drop_idx, *_ in matches:
            mol.RemoveAtom(drop_idx)
        error = Chem.SanitizeMol(mol, catchErrors=True)
    if error:
        raise ValueError(
            f"Monomer {monomer_name} not in monomer library and is not a valid SMILES string"
        )
    validate_monomer_core(monomer_name, mol)

    # Parsing without sanitizing records `/` and `\` as bond directions but never
    # works out the double bond geometry they describe, and sanitizing does not
    # do it either, so a monomer written with E/Z geometry would lose it.
    Chem.SetBondStereoFromDirections(mol)

    r_group_map = {}
    main_atoms = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        label = atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else ""

        if label.startswith("_R"):
            try:
                r_num = int(label[2:])
            except ValueError:
                raise ValueError(
                    f"Monomer {monomer_name} labels an atom {label}, which is not of the form _R<number>."
                ) from None
            if not 1 <= r_num <= MAX_RGROUPS:
                raise ValueError(
                    f"Monomer {monomer_name} labels an atom {label}; R-groups run from R1 to R{MAX_RGROUPS}."
                )
            if r_num in r_group_map:
                raise ValueError(
                    f"Monomer {monomer_name} labels more than one atom {label}."
                )
            atom.SetProp("dummyLabel", f"R{r_num}")
            atom.SetIntProp("_MolFileRLabel", r_num)
            atom.SetProp("molFileValue", "*")
            r_group_map[r_num] = idx
        else:
            main_atoms.append(idx)

    sorted_r = sorted(r_group_map.items())
    r_group_idx = [idx for _, idx in sorted_r]
    mol = Chem.RenumberAtoms(mol, main_atoms + r_group_idx)

    rgroup_idx_full: list[int | None] = [None] * MAX_RGROUPS
    for i, (r_num, _) in enumerate(sorted_r):
        rgroup_idx_full[r_num - 1] = len(main_atoms) + i

    attachment_points = infer_attachment_points(mol, rgroup_idx_full, monomer_name)
    rgroup_vals: list[str | None] = [None] * MAX_RGROUPS

    if m_type == "aa" and "_R1" not in monomer_name:
        matches = {
            idx
            for _, idx, _ in mol.GetSubstructMatches(
                Chem.MolFromSmarts("[#6][NX3H][#6]")
            )
        }
        if len(matches) == 0:
            matches = {
                idx
                for idx, _ in mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3H2][#6]"))
            }
        if len(matches) == 1:
            attachment_id = matches.pop()

            mol = Chem.RWMol(mol)
            new_idx = mol.AddAtom(Chem.Atom(0))
            mol.AddBond(attachment_id, new_idx, Chem.BondType.SINGLE)
            rgroup_idx_full[0] = new_idx
            attachment_points[0] = attachment_id

    if m_type == "aa" and "_R2" not in monomer_name:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3H1]=O"))
        hydroxyl = None
        if len(matches) == 0:
            acid = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OH]"))
            if len(acid) == 1:
                matches = acid
                hydroxyl = acid[0][2]
        if len(matches) == 1:
            attachment_id, *_ = matches[0]

            mol = Chem.RWMol(mol)
            if hydroxyl is None:
                new_idx = mol.AddAtom(Chem.Atom(0))
                mol.AddBond(attachment_id, new_idx, Chem.BondType.SINGLE)
            else:
                # The hydroxyl is the leaving group a peptide bond replaces, so
                # it becomes the R-group itself. Hanging a second atom off the
                # carboxyl carbon instead would give it five bonds as soon as
                # anything bonded through that R-group.
                mol.ReplaceAtom(hydroxyl, Chem.Atom(0))
                new_idx = hydroxyl
                rgroup_vals[1] = "OH"
            rgroup_idx_full[1] = new_idx
            attachment_points[1] = attachment_id

    # An amino acid caps an unused R2 with OH, the way every amino acid in the
    # monomer library does. Without it the carboxyl carbon keeps only its double
    # bonded oxygen once the dummy is deleted and the residue becomes an
    # aldehyde, so the same monomer spelled as SMILES and looked up by symbol
    # would not agree. Only an explicitly labelled R2 is capped: an R2 inferred
    # above sits on a carboxyl group that still carries its hydroxyl.
    if m_type == "aa" and "_R2" in monomer_name:
        r2_attachment = attachment_points[1]
        if r2_attachment is not None and _is_free_carbonyl_carbon(mol, r2_attachment):
            rgroup_vals[1] = "OH"

    mol.SetProp("symbol", monomer_name)
    mol.SetProp("m_abbr", monomer_name)
    mol.SetProp("m_type", m_type)
    mol.SetProp("m_RgroupIdx", ",".join(map(str, rgroup_idx_full)))
    mol.SetProp("m_Rgroups", ",".join(map(str, rgroup_vals)))
    mol.SetProp("m_attachmentPointIdx", ",".join(map(str, attachment_points)))
    mol.SetProp("natAnalog", "")

    return {
        "m_romol": mol,
        "m_Rgroups": rgroup_vals,
        "m_RgroupIdx": rgroup_idx_full,
        "m_attachmentPointIdx": attachment_points,
        "m_type": m_type,
        "m_abbr": monomer_name,
    }


_cap_group_re = re.compile(r"([A-Z][a-z]?)H?\d*")


@lru_cache
def _cap_group_atomic_number(cap_group: str) -> int | None:
    """Return the atomic number of the heavy atom of an R-group cap group.

    Cap groups are written as a condensed formula such as ``OH``, ``O`` or
    ``NH2``. Only the heavy atom has to be created: the hydrogens follow from
    the free valence once the molecule is sanitized. ``None`` is returned for
    cap groups that are not a single heavy atom.
    """
    match = _cap_group_re.fullmatch(cap_group)
    if not match:
        return None
    with rdBase.BlockLogs():
        try:
            return Chem.GetPeriodicTable().GetAtomicNumber(match.group(1))
        except RuntimeError:
            return None


class Molecule:
    """Single class for HELM to RDKit Mol conversion."""

    _bracket_re = re.compile(r"{(.*?)}")
    _annotation_re = re.compile(r'"[^"]*"')
    _rgroup_re = re.compile(r"R(\d+)")

    def __init__(self, helm: str, monomer_df: MonomerLibrary | None = None):
        """Initialize a Molecule object from a HELM string."""
        self._mol = None
        self.offset = []
        self.bondlist = []
        self.monomers = []
        self.chain_offset = {}
        self.residue_reps = defaultdict(list)
        self.has_ambiguous_monomers = False
        self.used_rgroups: set[tuple[int, int]] = set()
        self.hydrogen_bonds = []

        if monomer_df is None:
            self.monomer_df = load_monomer_library()
        else:
            self.monomer_df = monomer_df

        self._parse_helm_string(helm)
        self._build_molecule()

        if not isinstance(self._mol, Chem.rdchem.Mol):
            raise TypeError("Failed to initialize RDKit Mol object")

    @property
    def mol(self) -> Chem.Mol:
        assert self._mol is not None
        return self._mol

    def _parse_helm_string(self, helm: str) -> None:
        """Parse a HELM string into molecular components."""
        polymer_sections, connection_sections, hydrogen_bonds_sections, _, _ = (
            self._split_helm_sections(helm)
        )

        if not polymer_sections:
            raise ValueError(f"No simple polymers in HELM string {helm}")

        self._process_polymers(polymer_sections)
        self._process_connections(connection_sections)
        self._process_hydrogen_bonds(hydrogen_bonds_sections)

    @staticmethod
    def _split_outside_brackets(
        text: str, separator: str, maxsplit: int = 0
    ) -> list[str]:
        """Split on a separator that is not inside a bracketed monomer name.

        An inline SMILES monomer is written in square brackets and its CXSMILES
        part contains both separators, so the split tracks bracket depth rather
        than looking ahead for a closing bracket: a lookahead cannot tell a
        separator inside a monomer from one followed by a later section that
        happens to contain a bracket.
        """
        parts: list[str] = []
        current: list[str] = []
        depth = 0

        for char in text:
            if char == "[":
                depth += 1
            elif char == "]":
                depth = max(depth - 1, 0)

            if (
                char == separator
                and depth == 0
                and (not maxsplit or len(parts) < maxsplit)
            ):
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if depth:
            raise ValueError(f"Unbalanced brackets in {text}. Check HELM.")

        parts.append("".join(current))
        return parts

    def _split_helm_sections(
        self, helm: str
    ) -> tuple[list[str], list[str], list[str], str, str]:
        parts: list[str] = self._split_outside_brackets(helm, "$", 4)
        parts.extend([""] * (5 - len(parts)))

        polymers: list[str] = self._split_outside_brackets(parts[0], "|")
        connections = parts[1].split("|") if parts[1] else []
        hydrogen_bonds = parts[2].split("|") if parts[2] else []

        return polymers, connections, hydrogen_bonds, parts[3], parts[4]

    @staticmethod
    def _split_sequence_with_brackets(sequence: str) -> list[str]:
        """Split a sequence into individual monomers, respecting brackets."""
        result = []
        current = ""
        bracket_depth = 0

        for char in sequence:
            if char in "[(":
                bracket_depth += 1
                current += char
            elif char in "])":
                bracket_depth -= 1
                if bracket_depth < 0:
                    raise ValueError(
                        f"Unbalanced brackets in sequence {sequence}. Check HELM."
                    )
                current += char
            elif char == "." and bracket_depth == 0:
                result.append(current)
                current = ""
            else:
                current += char

        if bracket_depth:
            raise ValueError(f"Unbalanced brackets in sequence {sequence}. Check HELM.")

        # Appended even when empty: a trailing separator leaves a nameless
        # residue behind, which _process_monomer rejects rather than dropping.
        result.append(current)

        return result

    @staticmethod
    def _attachment_point(
        monomer: MonomerData, rgroup: int, position: int, context: str
    ) -> int:
        """Look up the atom a monomer's R-group bonds through.

        The R-group number is checked against the monomer rather than used as a
        list index: a monomer that declares fewer R-groups than the bond needs
        would otherwise come back as a bare IndexError.
        """
        attachment_points = monomer["m_attachmentPointIdx"]
        attachment = (
            attachment_points[rgroup] if 0 <= rgroup < len(attachment_points) else None
        )
        if attachment is None:
            raise ValueError(
                f"R-group {rgroup + 1} is not present in monomer {position} ({monomer['m_abbr']}). Check {context}."
            )
        return attachment

    @staticmethod
    def _extract_polymer_type(chain_str: str) -> Literal["PEPTIDE", "RNA", "CHEM"]:
        """Extract chain ID and return (chain_id, polymer_type)."""
        match = re.fullmatch(r"([A-Z]+)(\d+)", chain_str)
        if not match:
            raise ValueError(f"Invalid chain format: {chain_str}")

        polymer_type: str = match.group(1)
        if polymer_type not in {"PEPTIDE", "RNA", "CHEM"}:
            raise ValueError(f"Unsupported polymer type: {polymer_type}")

        return polymer_type

    def _process_monomer(
        self, monomer_name: str, chain_id: str, residue_idx: int, polymer_type: str
    ) -> MonomerData:
        """Process a single monomer."""
        bracketed = monomer_name.startswith("[") and monomer_name.endswith("]")
        if bracketed:
            monomer_name = monomer_name[1:-1]
        if not monomer_name:
            raise ValueError(f"Monomer {residue_idx + 1} has no name. Check HELM.")

        # Check for (a,[b]) pattern
        match = re.fullmatch(r"\([^,]+,\[([^\]]+)\]\)", monomer_name)
        if match:
            # Extract the 'b' from (a,[b]) and recurse
            self.has_ambiguous_monomers = True
            return self._process_monomer(
                f"[{match.group(1)}]", chain_id, residue_idx, polymer_type
            )

        if polymer_type == "PEPTIDE":
            m_type = "aa"
        elif polymer_type == "RNA":
            m_type = "rna"
        elif polymer_type == "CHEM":
            m_type = "chem"
        else:
            m_type = "aa"

        if m_type in self.monomer_df and monomer_name in self.monomer_df[m_type]:
            monomer_info = self.monomer_df[m_type][monomer_name]
        elif not bracketed:
            # HELM writes inline SMILES in square brackets, so a bare name is a
            # library symbol. Falling back to SMILES here would read a typo such
            # as PEPTIDE1{B} as a boron atom and build it without complaining.
            raise ValueError(
                f"Monomer {monomer_name} is not in the {m_type} monomer library. Inline SMILES has to be written in square brackets. Check HELM."
            )
        else:
            # Deliberately not written back into monomer_df: that dictionary is
            # shared by every Molecule through the cached monomer library, so
            # writing here would grow it with every inline monomer ever parsed.
            monomer_info = _create_missing_monomer(monomer_name, m_type)

        return {
            "m_romol": monomer_info["m_romol"],
            "m_Rgroups": monomer_info["m_Rgroups"][:],
            "m_RgroupIdx": monomer_info["m_RgroupIdx"],
            "m_attachmentPointIdx": monomer_info["m_attachmentPointIdx"],
            "m_type": monomer_info["m_type"],
            "m_abbr": monomer_info["m_abbr"],
        }

    @staticmethod
    def _parse_rna_string(sequence: str) -> list[str]:
        result = []
        current = ""
        bracket_depth = 0

        for char in sequence:
            if char in "[(":
                bracket_depth += 1
                current += char
            elif char in "])":
                bracket_depth -= 1
                current += char
            else:
                current += char
            if bracket_depth == 0:
                result.append(current)
                current = ""

        if current:
            result.append(current)

        return [r[1:-1] if r.startswith("[") and r.endswith("]") else r for r in result]

    def _process_polymers(self, polymers: list[str]) -> None:
        """Process polymer chains from HELM, creating backbone bonds on the fly."""
        monomer_idx = 0

        for chain in polymers:
            chain = chain.strip()
            match = self._bracket_re.search(chain)
            if not match:
                raise ValueError(
                    f"Polymer {chain} is not of the form CHAIN{{sequence}}. Check HELM."
                )

            trailing = chain[match.end() :]
            if trailing and not self._annotation_re.fullmatch(trailing):
                raise ValueError(
                    f"Unexpected text {trailing} after the sequence of polymer {chain}. Check HELM."
                )

            chain_id = chain[: match.start()]
            polymer_type = self._extract_polymer_type(chain_id)

            if chain_id in self.chain_offset:
                raise ValueError(f"Duplicate chain ID: {chain_id}")

            sequence = match.group(1)
            if not sequence:
                raise ValueError(
                    f"Polymer {chain_id} has an empty sequence. Check HELM."
                )

            residues = self._split_sequence_with_brackets(sequence)
            self.chain_offset[chain_id] = monomer_idx

            if polymer_type == "PEPTIDE":
                for residue_idx, monomer_name in enumerate(residues):
                    monomer = self._process_monomer(
                        monomer_name, chain_id, residue_idx, polymer_type
                    )

                    self.monomers.append(monomer)
                    self.residue_reps[chain_id].append(monomer_idx)

                    if residue_idx > 0:
                        monomer1 = self.monomers[monomer_idx - 1]
                        monomer2 = monomer

                        attachment_point1 = self._attachment_point(
                            monomer1, 1, monomer_idx, "monomers"
                        )
                        attachment_point2 = self._attachment_point(
                            monomer2, 0, monomer_idx + 1, "monomers"
                        )

                        self.bondlist.append([
                            monomer_idx - 1,
                            attachment_point1,
                            monomer_idx,
                            attachment_point2,
                        ])
                        self._mark_used_rgroup(monomer_idx - 1, 1)
                        self._mark_used_rgroup(monomer_idx, 0)

                    monomer_idx += 1
            elif polymer_type == "RNA":
                prev_monomer = None
                for residue_idx, residue in enumerate(residues):
                    split_residue = self._parse_rna_string(residue)
                    for subresidue in split_residue:
                        is_base = subresidue.startswith("(") and subresidue.endswith(
                            ")"
                        )
                        monomer_name = subresidue[1:-1] if is_base else subresidue
                        if is_base and prev_monomer is None:
                            raise ValueError(
                                f"Branch monomer ({monomer_name}) starts chain {chain_id} and has nothing to attach to. Check HELM."
                            )

                        monomer = self._process_monomer(
                            monomer_name, chain_id, residue_idx, polymer_type
                        )

                        self.monomers.append(monomer)
                        self.residue_reps[chain_id].append(monomer_idx)

                        if prev_monomer is not None:
                            monomer1 = self.monomers[prev_monomer]
                            monomer2 = monomer

                            # Attach to R3 to R1 if the monomer is a base, R2 to R1 otherwise
                            r_index = 2 if is_base else 1
                            attachment_point1 = self._attachment_point(
                                monomer1, r_index, prev_monomer + 1, "monomers"
                            )
                            attachment_point2 = self._attachment_point(
                                monomer2, 0, monomer_idx + 1, "monomers"
                            )

                            self.bondlist.append([
                                prev_monomer,
                                attachment_point1,
                                monomer_idx,
                                attachment_point2,
                            ])
                            self._mark_used_rgroup(prev_monomer, r_index)
                            self._mark_used_rgroup(monomer_idx, 0)

                        # Only set prev_monomer if the monomer is not a base
                        if not is_base:
                            prev_monomer = monomer_idx

                        monomer_idx += 1
            elif polymer_type == "CHEM":
                if len(residues) != 1:
                    raise ValueError("CHEM polymers must have exactly one residue")
                monomer_name = residues[0]
                residue_idx = 0
                monomer = self._process_monomer(
                    monomer_name, chain_id, residue_idx, polymer_type
                )
                self.monomers.append(monomer)
                self.residue_reps[chain_id].append(monomer_idx)
                monomer_idx += 1
            else:
                assert_never(polymer_type)

    @staticmethod
    def _parse_residue_number(value: str, bond_spec: str) -> int:
        """Convert a 1-based residue number from a bond specification."""
        try:
            residue = int(value)
        except ValueError:
            raise ValueError(
                f"Residue number {value} in {bond_spec} is not a number. Check HELM."
            ) from None
        if residue < 1:
            raise ValueError(
                f"Residue number {value} in {bond_spec} is not positive; residues are numbered from 1. Check HELM."
            )
        return residue - 1

    @staticmethod
    def _parse_rgroup_number(value: str, bond_spec: str) -> int:
        """Convert an R-group label such as ``R3`` from a bond specification."""
        match = Molecule._rgroup_re.fullmatch(value)
        if not match:
            raise ValueError(
                f"R-group {value} in {bond_spec} is not of the form R<number>. Check HELM."
            )
        rgroup = int(match.group(1))
        if rgroup < 1:
            raise ValueError(
                f"R-group {value} in {bond_spec} is not positive; R-groups are numbered from 1. Check HELM."
            )
        return rgroup

    @staticmethod
    def _parse_connection(connection_str: str) -> tuple[str, int, int, str, int, int]:
        """Parse a single connection string.

        A connection that cannot be parsed is an error rather than a warning:
        skipping it would return a molecule that is quietly missing a bond.
        """
        parts = connection_str.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid connection format: {connection_str}. Check HELM."
            )

        chain_id1, chain_id2, bond_spec = parts

        bond_parts = re.split(r"[-:]", bond_spec)
        if len(bond_parts) != 4:
            raise ValueError(f"Invalid bond format: {bond_spec}. Check HELM.")

        residue1, rgroup1, residue2, rgroup2 = bond_parts

        return (
            chain_id1,
            Molecule._parse_residue_number(residue1, bond_spec),
            Molecule._parse_rgroup_number(rgroup1, bond_spec),
            chain_id2,
            Molecule._parse_residue_number(residue2, bond_spec),
            Molecule._parse_rgroup_number(rgroup2, bond_spec),
        )

    def _resolve_residue(self, chain_id: str, residue: int, context: str) -> int:
        """Look up the monomer index of a 0-based residue of a declared chain."""
        residues = self.residue_reps.get(chain_id)
        if not residues:
            raise ValueError(
                f"Chain {chain_id} is not a polymer in this HELM string. Check {context}."
            )
        if not 0 <= residue < len(residues):
            raise ValueError(
                f"Residue {residue + 1} is out of range for chain {chain_id}, which has {len(residues)} residues. Check {context}."
            )
        return residues[residue]

    def _resolve_connection_endpoint(
        self, chain_id: str, residue: int, rgroup: int
    ) -> tuple[int, int]:
        """Resolve one side of a connection to (monomer index, attachment atom).

        The R-group number is checked against the monomer rather than used as a
        list index: numbering starts at one, so anything out of range would
        otherwise be read as a negative index and silently bond the wrong atoms.
        """
        monomer_idx = self._resolve_residue(chain_id, residue, "connections")
        attachment_idx = self._attachment_point(
            self.monomers[monomer_idx], rgroup, monomer_idx + 1, "connections"
        )

        return monomer_idx, attachment_idx

    def _process_connections(self, connections: list[str]) -> None:
        """Process connections between chains."""
        if not connections:
            return

        for connection_str in connections:
            chain_id1, residue1, rgroup1, chain_id2, residue2, rgroup2 = (
                self._parse_connection(connection_str)
            )

            monomer_idx1, attachment_idx1 = self._resolve_connection_endpoint(
                chain_id1, residue1, rgroup1 - 1
            )
            monomer_idx2, attachment_idx2 = self._resolve_connection_endpoint(
                chain_id2, residue2, rgroup2 - 1
            )

            self.bondlist.append([
                monomer_idx1,
                attachment_idx1,
                monomer_idx2,
                attachment_idx2,
            ])

            self._mark_used_rgroup(monomer_idx1, rgroup1 - 1)
            self._mark_used_rgroup(monomer_idx2, rgroup2 - 1)

    def _process_hydrogen_bonds(self, connections: list[str]) -> None:
        """Process hydrogen bonds."""
        if not connections:
            return

        for connection_str in connections:
            parts = connection_str.split(",")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid hydrogen bond format: {connection_str}. Check HELM."
                )
            chain_id1, chain_id2, bond_spec = parts

            bond_parts = re.split(r"[-:]", bond_spec)
            if len(bond_parts) != 4:
                raise ValueError(
                    f"Invalid hydrogen bond format: {bond_spec}. Check HELM."
                )

            residue1 = self._parse_residue_number(bond_parts[0], bond_spec)
            residue2 = self._parse_residue_number(bond_parts[2], bond_spec)
            self._resolve_residue(chain_id1, residue1, "hydrogen bonds")
            self._resolve_residue(chain_id2, residue2, "hydrogen bonds")
            self.hydrogen_bonds.append([chain_id1, residue1, chain_id2, residue2])

    def _mark_used_rgroup(self, monomer_idx: int, rgroup: int) -> None:
        """Claim an R-group for a bond, refusing one that is already spent.

        An R-group stands for one attachment. Letting a second bond claim it
        puts both bonds on the same atom, which overfills it and hands back a
        molecule RDKit cannot even read back from its own SMILES.
        """
        monomer = self.monomers[monomer_idx]
        if (monomer_idx, rgroup) in self.used_rgroups:
            raise ValueError(
                f"R-group {rgroup + 1} of monomer {monomer_idx + 1} ({monomer['m_abbr']}) is bonded more than once. Check HELM."
            )
        self.used_rgroups.add((monomer_idx, rgroup))
        monomer["m_Rgroups"][rgroup] = None

    def _build_molecule(self) -> None:
        """Build the RDKit molecule from parsed monomer and bond data."""
        if not self.monomers:
            self._mol = Chem.RWMol()
            return

        monomer = self.monomers[0]
        self._mol = Chem.RWMol(monomer["m_romol"])

        self._replace_rgroups(0, monomer)

        current_offset = self._mol.GetNumAtoms()
        self.offset = [0, current_offset]

        for monomer in self.monomers[1:]:
            self._mol.InsertMol(monomer["m_romol"])

            self._replace_rgroups(current_offset, monomer)

            atom_count = monomer["m_romol"].GetNumAtoms()
            current_offset += atom_count
            self.offset.append(current_offset)

        self._add_bonds()
        self._sanitize()

    def _add_bonds(self) -> None:
        """Add bonds between monomers based on bond list."""
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self.bondlist:
            absolute_atom1_idx = self.offset[monomer1_idx] + atom1_idx
            absolute_atom2_idx = self.offset[monomer2_idx] + atom2_idx

            # RDKit answers both of these with a C++ pre-condition violation,
            # which tells a caller nothing about which connection is at fault.
            if absolute_atom1_idx == absolute_atom2_idx:
                raise ValueError(
                    f"Monomer {monomer1_idx + 1} is bonded to itself through one atom. Check connections."
                )
            if (
                self.mol.GetBondBetweenAtoms(absolute_atom1_idx, absolute_atom2_idx)
                is not None
            ):
                raise ValueError(
                    f"Duplicate bond between monomer {monomer1_idx + 1} and monomer {monomer2_idx + 1}. Check connections."
                )

            self.mol.AddBond(
                absolute_atom1_idx, absolute_atom2_idx, Chem.BondType.SINGLE
            )

    def _replace_rgroups(self, atom_offset: int, monomer: MonomerData) -> None:
        """Cap every R-group of a monomer that no bond has claimed."""
        for rgroup, (cap, atom_idx) in enumerate(
            zip(monomer["m_Rgroups"], monomer["m_RgroupIdx"]), start=1
        ):
            if cap is None:
                continue
            if atom_idx is None:
                raise ValueError(
                    f"R-group {rgroup} of monomer {monomer['m_abbr']} has the cap group {cap} but no atom index."
                )
            self._replace_rgroup(atom_offset, atom_idx, cap)

    def _replace_rgroup(self, atom_offset: int, atom_idx: int, atom_type: str) -> None:
        """Replace an unused R-group with the heavy atom of its cap group."""
        # A hydrogen cap needs no atom of its own: the dummy is dropped while
        # sanitizing and RDKit fills the free valence with an implicit hydrogen.
        if atom_type == "H":
            return

        # Leaving the cap group off would delete the dummy atom along with the
        # used R-groups and hand back a molecule that is quietly missing an atom.
        atomic_number = _cap_group_atomic_number(atom_type)
        if atomic_number is None:
            raise ValueError(
                f"Unsupported R-group cap group {atom_type}. Check monomers."
            )

        try:
            self.mol.ReplaceAtom(atom_offset + atom_idx, Chem.Atom(atomic_number))
        except (RuntimeError, OverflowError) as e:
            raise ValueError(
                f"Failed to replace R-group with {atom_type}: {e}. Check monomers."
            ) from e

    def _move_stereo_references(self, deleted: set[int]) -> None:
        """Move double bond stereo references off the atoms about to be deleted.

        RDKit drops a reference that no longer exists, and a double bond with no
        references loses the geometry the monomer declared. The atom an
        inter-monomer bond was made to stands where the R-group stood, so it
        takes the reference over unchanged; any other surviving neighbour is on
        the opposite side of the double bond and flips the parity.
        """
        bonded: defaultdict[int, set[int]] = defaultdict(set)
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self.bondlist:
            first = self.offset[monomer1_idx] + atom1_idx
            second = self.offset[monomer2_idx] + atom2_idx
            bonded[first].add(second)
            bonded[second].add(first)

        for bond in self.mol.GetBonds():
            if bond.GetStereo() not in _FLIPPED_STEREO:
                continue
            references = list(bond.GetStereoAtoms())
            if len(references) != 2 or not deleted.intersection(references):
                continue

            ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            moved: list[int] = []
            flip = False

            for reference in references:
                if reference not in deleted:
                    moved.append(reference)
                    continue
                end = next(
                    (e for e in ends if self.mol.GetBondBetweenAtoms(e, reference)),
                    None,
                )
                if end is None:
                    break
                other_end = ends[1] if end == ends[0] else ends[0]
                candidates = [
                    neighbour.GetIdx()
                    for neighbour in self.mol.GetAtomWithIdx(end).GetNeighbors()
                    if neighbour.GetIdx() not in deleted
                    and neighbour.GetIdx() != other_end
                ]
                if len(candidates) != 1:
                    break
                if candidates[0] not in bonded[end]:
                    flip = not flip
                moved.append(candidates[0])

            if len(moved) != 2:
                # Nothing dependable to point at, so drop the geometry rather
                # than state one that might be the wrong way round.
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                continue

            bond.SetStereoAtoms(*moved)
            if flip:
                bond.SetStereo(_FLIPPED_STEREO[bond.GetStereo()])

    def _sanitize(self) -> None:
        """Clean up the molecule by removing dummy atoms."""
        pattern = Chem.MolFromSmarts("[#0]")
        matches = self.mol.GetSubstructMatches(pattern)
        atoms_to_delete = sorted({idx for match in matches for idx in match})
        self._move_stereo_references(set(atoms_to_delete))
        self._mol = Chem.DeleteSubstructs(self._mol, pattern)

        def correction(offset: int, idx: int) -> int:
            return bisect.bisect_left(atoms_to_delete, idx) - bisect.bisect_left(
                atoms_to_delete, offset
            )

        for i, (m1, a1, m2, a2) in enumerate(self.bondlist):
            offset1 = self.offset[m1]
            offset2 = self.offset[m2]
            self.bondlist[i][1] -= correction(offset1, offset1 + a1)
            self.bondlist[i][3] -= correction(offset2, offset2 + a2)

        self.offset = [
            offset - sum(d < offset for d in atoms_to_delete) for offset in self.offset
        ]

        # Ring membership is not worked out by any of the above, and RDKit
        # answers a ring query on a molecule without it by raising, which takes
        # out ring descriptors and every SMARTS match that mentions a ring.
        Chem.FastFindRings(self._mol)

        # Stereo is carried on the double bond, but writing SMILES needs the
        # direction of the single bonds around it, which nothing above sets. A
        # molecule would report its geometry through InChI and lose it through
        # SMILES.
        Chem.SetDoubleBondNeighborDirections(self._mol)

    @property
    def bond_indices(self) -> list[int]:
        indices = []
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self.bondlist:
            bond = self.mol.GetBondBetweenAtoms(
                self.offset[monomer1_idx] + atom1_idx,
                self.offset[monomer2_idx] + atom2_idx,
            )
            if bond is None:
                raise ValueError(
                    f"The bond between monomer {monomer1_idx + 1} and monomer {monomer2_idx + 1} is not present in the molecule."
                )
            indices.append(bond.GetIdx())
        return indices

    @property
    def monomer_indices(self) -> list[int]:
        return [
            bisect.bisect_right(self.offset, i) - 1
            for i in range(self.mol.GetNumAtoms())
        ]


_monomer_df: MonomerLibrary | None = None


def _init_pool(monomer_df: MonomerLibrary):
    global _monomer_df
    _monomer_df = monomer_df


def _load_helm(helm: str) -> Molecule:
    return Molecule(helm, _monomer_df)


def load_in_parallel(
    helms: list[str],
    monomer_df: MonomerLibrary | None = None,
    chunksize: int | None = 256,
) -> list[Molecule]:
    if monomer_df is None:
        monomer_df = load_monomer_library()
    with multiprocessing.Pool(initializer=_init_pool, initargs=(monomer_df,)) as pool:
        return pool.map(_load_helm, helms, chunksize=chunksize)
