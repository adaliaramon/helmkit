import bisect
import itertools
import multiprocessing
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
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

PolymerType = Literal["PEPTIDE", "RNA", "CHEM"]

# The monomer library type that the monomers of each polymer type come from.
_MONOMER_TYPES: dict[PolymerType, str] = {"PEPTIDE": "aa", "RNA": "rna", "CHEM": "chem"}

_FLIPPED_STEREO = {
    Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS,
}

# A carbon bonded to one oxygen by a double bond, another by a single bond and
# two further neighbours: a carboxyl carbon written with one bond too many.
_OVERVALENT_CARBOXYL = Chem.MolFromSmarts("O[CX4]=O")
_BACKBONE_AMINE = Chem.MolFromSmarts("[NX3;H1,H2][CX4][CX3]=O")
_SECONDARY_AMINE = Chem.MolFromSmarts("[#6][NX3H][#6]")
_PRIMARY_AMINE = Chem.MolFromSmarts("[NX3H2][#6]")
_ALDEHYDE = Chem.MolFromSmarts("[CX3H1]=O")
_CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=O)[OH]")
_BACKBONE_CARBONYL = Chem.MolFromSmarts("[#7][CX4][CX3]=O")
_DUMMY = Chem.MolFromSmarts("[#0]")
# A dummy atom bonded to one end of a double bond, as (dummy, end, other end).
# The other end needs a neighbour of its own for the bond to have E/Z geometry,
# which leaves out the carbonyl next to every R2.
_DUMMY_BY_DOUBLE_BOND = Chem.MolFromSmarts("[#0]~*=[!D1]")

# `GetSubstructMatches` stops at 1000 matches unless told otherwise, which
# would miss atoms in a long polymer.
_ALL_MATCHES = 2**32 - 1

# The characters that matter when splitting a sequence into monomers.
_SEQUENCE_SPECIAL = frozenset('"[]().')


def get_molecule_property(
    molecule: Chem.Mol, property_name: str, default: str | None = None
) -> str | None:
    # `Mol.GetProp` only accepts a `default` argument in recent RDKit releases.
    if not molecule.HasProp(property_name):
        return default
    return molecule.GetProp(property_name)


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
        return [None if v == "None" else convert_func(v) for v in values]
    return [None if v == "None" else v for v in values]


def _dummy_atoms(molecule: Chem.Mol) -> list[int]:
    """Return the indices of every dummy atom, in ascending order."""
    matches = molecule.GetSubstructMatches(_DUMMY, maxMatches=_ALL_MATCHES)
    return sorted(idx for (idx,) in matches)


def _delete_atoms(molecule: Chem.RWMol, indices: Iterable[int]) -> None:
    """Delete atoms in place, the way `Chem.DeleteSubstructs` does."""
    molecule.BeginBatchEdit()
    for idx in indices:
        molecule.RemoveAtom(idx)
    molecule.CommitBatchEdit()
    molecule.ClearComputedProps()
    molecule.UpdatePropertyCache(strict=False)


def infer_attachment_points(
    molecule: Chem.Mol, rgroup_indices: Sequence[int | None], name: str = "monomer"
) -> list[int | None]:
    """Infer attachment points by finding atoms bonded to R-group atoms."""
    attachment_points: list[int | None] = []

    for rgroup, r_idx in enumerate(rgroup_indices, start=1):
        if r_idx is None:
            attachment_points.append(None)
            continue

        # The attachment point has to be a real atom. Dummy atoms are all
        # deleted while sanitizing, so an R-group bonded only to other dummies
        # would leave the bonds made to it pointing at atoms that are gone and
        # drop the monomer out of the molecule without a word.
        for neighbour in molecule.GetAtomWithIdx(r_idx).GetNeighbors():
            if neighbour.GetAtomicNum() != 0:
                attachment_points.append(neighbour.GetIdx())
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

    A salt may carry its counter-ion as a fragment of its own, such as the
    sodium of ``[Na+].[O-]P(*)(*)=O``. Only a charged fragment with no R-group
    can stand apart that way: anything else is a second piece of the monomer
    that no bond would ever join to the rest.
    """
    core = Chem.RWMol(molecule)
    dummies = _dummy_atoms(core)
    _delete_atoms(core, dummies)
    if core.GetNumAtoms() == 0:
        raise ValueError(f"Monomer {name} has no atoms besides its R-groups.")

    fragments = Chem.GetMolFrags(core)
    if len(fragments) == 1:
        return

    # Where each atom bonded to an R-group ended up once the dummies went.
    carries_rgroup = {
        neighbour.GetIdx() - bisect.bisect_left(dummies, neighbour.GetIdx())
        for idx in dummies
        for neighbour in molecule.GetAtomWithIdx(idx).GetNeighbors()
        if neighbour.GetAtomicNum() != 0
    }
    pieces = 0
    for fragment in fragments:
        charge = sum(core.GetAtomWithIdx(idx).GetFormalCharge() for idx in fragment)
        counter_ion = charge != 0 and carries_rgroup.isdisjoint(fragment)
        pieces += not counter_ion
    if pieces > 1:
        raise ValueError(
            f"Monomer {name} falls into separate fragments once its R-groups are removed, and only a charged counter-ion can stand apart."
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

        m_type = get_molecule_property(mol, "m_type") or ""
        if m_type not in _MONOMER_TYPES.values():
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

        monomers_dict[m_type][symbol] = {
            "m_romol": mol,
            "m_Rgroups": rgroups,
            "m_RgroupIdx": rgroup_idx,
            "m_attachmentPointIdx": infer_attachment_points(mol, rgroup_idx, symbol),
            "m_type": m_type,
            # m_abbr is only used for display, so fall back to the symbol when
            # a library does not provide it instead of dropping the monomer.
            "m_abbr": get_molecule_property(mol, "m_abbr") or symbol,
        }

    return monomers_dict


def _is_free_carbonyl_carbon(molecule: Chem.Mol, idx: int) -> bool:
    """Is this a carbonyl carbon that does not already carry a second oxygen?"""
    atom = molecule.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return False

    carbonyl = False
    for bond in atom.GetBonds():
        if bond.GetOtherAtom(atom).GetAtomicNum() != 8:
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            return False
        carbonyl = True
    return carbonyl


def _sanitize_inline_monomer(monomer_name: str, mol: Chem.Mol) -> Chem.Mol:
    """Sanitize a monomer read from inline SMILES without sanitizing."""
    with rdBase.BlockLogs():
        error = Chem.SanitizeMol(mol, catchErrors=True)
        if error == Chem.SanitizeFlags.SANITIZE_PROPERTIES:
            mol = Chem.RWMol(mol)
            matches = mol.GetSubstructMatches(_OVERVALENT_CARBOXYL)
            # Removing an atom shifts every index above it down by one, so the
            # indices are taken highest first. Working upwards would delete
            # whatever had moved into the place of the second match.
            for drop_idx in sorted({match[0] for match in matches}, reverse=True):
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
    return mol


def _rgroups_from_atom_maps(monomer_name: str, mol: Chem.Mol) -> dict[int, str | None]:
    """Turn the atom-mapped leaving atoms of HELM2 inline SMILES into R-groups.

    HELM2 can mark R-group ``n`` with atom map ``n`` on the atom that leaves
    when the bond is made: ``[*:1]`` is a bare R-group, while ``[H:1]`` and
    ``[OH:2]`` also say what caps it when it is not used. Each becomes a dummy
    labelled ``_R<n>``, the same as the CXSMILES spelling. Returns the cap of
    each R-group found, by number.
    """
    caps: dict[int, str | None] = {}
    for atom in mol.GetAtoms():
        r_num = atom.GetAtomMapNum()
        if not r_num:
            continue
        if atom.HasProp("atomLabel"):
            raise ValueError(
                f"Monomer {monomer_name} gives atom {atom.GetIdx()} both an atom map and an atom label; mark each R-group one way."
            )
        bonds = atom.GetBonds()
        if len(bonds) != 1 or bonds[0].GetBondType() != Chem.BondType.SINGLE:
            raise ValueError(
                f"Monomer {monomer_name} maps atom {atom.GetIdx()} as R{r_num}, but only an atom joined to the monomer by a single bond can be an R-group."
            )
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons():
            raise ValueError(
                f"Monomer {monomer_name} maps a charged, isotopic or radical atom as R{r_num}, which cannot be a cap group."
            )
        if r_num > MAX_RGROUPS:
            raise ValueError(
                f"Monomer {monomer_name} maps an atom as R{r_num}; R-groups run from R1 to R{MAX_RGROUPS}."
            )
        if r_num in caps:
            raise ValueError(
                f"Monomer {monomer_name} maps more than one atom as R{r_num}."
            )

        if atom.GetAtomicNum() == 0:
            caps[r_num] = None
        else:
            hydrogens = atom.GetNumExplicitHs()
            caps[r_num] = (
                atom.GetSymbol()
                + "H" * bool(hydrogens)
                + (str(hydrogens) if hydrogens > 1 else "")
            )

        atom.SetAtomicNum(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetAtomMapNum(0)
        atom.SetProp("atomLabel", f"_R{r_num}")
    return caps


def _number_rgroups(
    monomer_name: str, mol: Chem.Mol
) -> tuple[Chem.RWMol, list[int | None]]:
    """Mark the atoms labelled ``_R<n>`` as R-groups and move them to the end.

    Returns the renumbered molecule and the index of each R-group atom, with
    ``None`` for the R-groups the monomer does not have.
    """
    rgroup_atoms: dict[int, int] = {}
    core_atoms: list[int] = []

    for atom in mol.GetAtoms():
        label = atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else ""
        if not label.startswith("_R"):
            # Every dummy atom is deleted once the molecule is built, so one
            # that is not an R-group would vanish along with whatever cap or
            # bond it was meant to stand for.
            if atom.GetAtomicNum() == 0:
                raise ValueError(
                    f"Monomer {monomer_name} has a dummy atom (atom {atom.GetIdx()}) with no _R<number> label or atom map, so it is not an R-group."
                )
            core_atoms.append(atom.GetIdx())
            continue

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
        if r_num in rgroup_atoms:
            raise ValueError(
                f"Monomer {monomer_name} labels more than one atom {label}."
            )
        # Only dummy atoms are deleted once the molecule is built, so a label
        # on a real atom would keep that atom as well as the bond made in its
        # place, while the dummy it was meant for disappears.
        if atom.GetAtomicNum() != 0:
            raise ValueError(
                f"Monomer {monomer_name} labels a {atom.GetSymbol()} atom {label}; only a dummy atom (*) can be an R-group."
            )
        atom.SetProp("dummyLabel", f"R{r_num}")
        atom.SetIntProp("_MolFileRLabel", r_num)
        atom.SetProp("molFileValue", "*")
        rgroup_atoms[r_num] = atom.GetIdx()

    rgroups = sorted(rgroup_atoms)
    mol = Chem.RenumberAtoms(mol, core_atoms + [rgroup_atoms[r] for r in rgroups])

    rgroup_idx: list[int | None] = [None] * MAX_RGROUPS
    for new_idx, r_num in enumerate(rgroups, start=len(core_atoms)):
        rgroup_idx[r_num - 1] = new_idx
    return Chem.RWMol(mol), rgroup_idx


def _add_amine_rgroup(mol: Chem.RWMol) -> tuple[int, int] | None:
    """Give an amino acid an R1 on its backbone amine.

    Returns the new R-group atom and the amine it is bonded to, or ``None``
    when there is no single amine to choose.
    """
    # The amine a peptide bond is made to is the one on the alpha carbon, the
    # carbon that also carries the carboxyl. Asking only for a secondary amine
    # finds a side chain amine just as readily as the backbone one, and asking
    # only for a primary amine misses a substituted backbone.
    amines = {m[0] for m in mol.GetSubstructMatches(_BACKBONE_AMINE)}
    if len(amines) != 1:
        amines = {m[1] for m in mol.GetSubstructMatches(_SECONDARY_AMINE)}
    if not amines:
        amines = {m[0] for m in mol.GetSubstructMatches(_PRIMARY_AMINE)}
    if len(amines) != 1:
        return None

    (amine,) = amines
    rgroup = mol.AddAtom(Chem.Atom(0))
    mol.AddBond(amine, rgroup, Chem.BondType.SINGLE)
    return rgroup, amine


def _add_carboxyl_rgroup(mol: Chem.RWMol) -> tuple[int, int, str | None] | None:
    """Give an amino acid an R2 on its backbone carbonyl.

    Returns the R-group atom, the carbonyl carbon it is bonded to and the cap
    group it replaced, or ``None`` when there is no single carbonyl to choose.
    """
    aldehydes = {m[0] for m in mol.GetSubstructMatches(_ALDEHYDE)}
    acids = {m[0]: m[2] for m in mol.GetSubstructMatches(_CARBOXYLIC_ACID)}
    # The carboxyl a peptide bond is made to is the one on the alpha carbon, the
    # carbon that also carries the amine. Asking only for an aldehyde or only
    # for an acid finds a side chain carbonyl just as readily.
    on_alpha = {m[2] for m in mol.GetSubstructMatches(_BACKBONE_CARBONYL)} & (
        aldehydes | acids.keys()
    )
    carbonyls = on_alpha if len(on_alpha) == 1 else aldehydes or set(acids)
    if len(carbonyls) != 1:
        return None

    (carbonyl,) = carbonyls
    hydroxyl = acids.get(carbonyl)
    if hydroxyl is None:
        rgroup = mol.AddAtom(Chem.Atom(0))
        mol.AddBond(carbonyl, rgroup, Chem.BondType.SINGLE)
        return rgroup, carbonyl, None

    # The hydroxyl is the leaving group a peptide bond replaces, so it becomes
    # the R-group itself. Hanging a second atom off the carboxyl carbon instead
    # would give it five bonds as soon as anything bonded through that R-group.
    mol.ReplaceAtom(hydroxyl, Chem.Atom(0))
    return hydroxyl, carbonyl, "OH"


def _create_missing_monomer(monomer_name: str, m_type: str = "aa") -> MonomerData:
    """Build a monomer from an inline CXSMILES string with ``_R<n>`` labels."""
    # The name may just be a symbol the library lacks, such as [Dig]; the error
    # below says so, and RDKit's own account of the parse would only add noise.
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(monomer_name, sanitize=False)
    if mol is None:
        # Accept CXSMILES atom labels whose closing `$` is missing.
        if monomer_name.endswith("|") and not monomer_name.endswith("$|"):
            return _create_missing_monomer(monomer_name[:-1] + "$|", m_type)
        raise ValueError(
            f"Monomer {monomer_name} not in monomer library and is not a valid SMILES string"
        )

    mapped_caps = _rgroups_from_atom_maps(monomer_name, mol)
    mol = _sanitize_inline_monomer(monomer_name, mol)
    mol, rgroup_idx = _number_rgroups(monomer_name, mol)
    attachment_points = infer_attachment_points(mol, rgroup_idx, monomer_name)
    caps: list[str | None] = [None] * MAX_RGROUPS
    for r_num, cap in mapped_caps.items():
        caps[r_num - 1] = cap

    if m_type == "aa":
        has_r1, has_r2 = rgroup_idx[0] is not None, rgroup_idx[1] is not None

        if not has_r1 and (r1 := _add_amine_rgroup(mol)):
            rgroup_idx[0], attachment_points[0] = r1

        if not has_r2 and (r2 := _add_carboxyl_rgroup(mol)):
            rgroup_idx[1], attachment_points[1], caps[1] = r2

        # An amino acid caps an unused R2 with OH, the way every amino acid in
        # the monomer library does. Without it the carboxyl carbon keeps only
        # its double bonded oxygen once the dummy is deleted and the residue
        # becomes an aldehyde, so the same monomer spelled as SMILES and looked
        # up by symbol would not agree. Only an explicitly labelled R2 is
        # capped: an R2 inferred above sits on a carboxyl group that still
        # carries its hydroxyl. A cap the SMILES spells out is kept.
        r2_attachment = attachment_points[1]
        if (
            has_r2
            and caps[1] is None
            and r2_attachment is not None
            and _is_free_carbonyl_carbon(mol, r2_attachment)
        ):
            caps[1] = "OH"

    # Inferring an R-group adds or replaces atoms, which leaves the hydrogen
    # counts RDKit keeps for the atoms around it out of date.
    mol.UpdatePropertyCache(strict=False)

    mol.SetProp("symbol", monomer_name)
    mol.SetProp("m_abbr", monomer_name)
    mol.SetProp("m_type", m_type)
    mol.SetProp("m_RgroupIdx", ",".join(map(str, rgroup_idx)))
    mol.SetProp("m_Rgroups", ",".join(map(str, caps)))
    mol.SetProp("m_attachmentPointIdx", ",".join(map(str, attachment_points)))
    mol.SetProp("natAnalog", "")

    return {
        "m_romol": mol,
        "m_Rgroups": caps,
        "m_RgroupIdx": rgroup_idx,
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

    _annotation_re = re.compile(r'"[^"]*"')
    _polymer_group_re = re.compile(r"G\d+\(.*\)")
    _chain_id_re = re.compile(r"([A-Z]+)(\d+)")
    _ambiguous_re = re.compile(r"\([^,]+,\[([^\]]+)\]\)")
    _bond_spec_re = re.compile(r"[-:]")
    _rgroup_re = re.compile(r"R(\d+)")

    def __init__(self, helm: str, monomer_df: MonomerLibrary | None = None):
        """Initialize a Molecule object from a HELM string."""
        self.monomer_df = load_monomer_library() if monomer_df is None else monomer_df
        self.monomers: list[MonomerData] = []
        # Each inter-monomer bond as [monomer1, atom1, monomer2, atom2], with
        # atom indices relative to their monomer.
        self.bondlist: list[list[int]] = []
        # The 0-based R-groups each bond in the bond list was made through.
        self.bond_rgroups: list[tuple[int, int]] = []
        # The index of the first atom of each monomer, followed by the total.
        self.offset: list[int] = []
        self.chain_offset: dict[str, int] = {}
        self.residue_reps: defaultdict[str, list[int]] = defaultdict(list)
        self.has_ambiguous_monomers = False
        self.used_rgroups: set[tuple[int, int]] = set()
        self.hydrogen_bonds: list[list[str | int]] = []

        self._parse_helm_string(helm)
        self.mol: Chem.Mol = self._build_molecule()

    # Parsing

    def _parse_helm_string(self, helm: str) -> None:
        """Parse a HELM string into monomers and the bonds between them."""
        polymers, connections, hydrogen_bonds = self._split_helm_sections(helm)
        self._process_polymers(polymers)
        self._process_connections(connections)
        self._process_hydrogen_bonds(hydrogen_bonds)

    @staticmethod
    def _split_outside_brackets(
        text: str, separator: str, maxsplit: int = 0
    ) -> list[str]:
        """Split on a separator that is not inside a bracketed monomer name.

        An inline SMILES monomer is written in square brackets and its CXSMILES
        part contains both separators, so the split tracks bracket depth rather
        than looking ahead for a closing bracket: a lookahead cannot tell a
        separator inside a monomer from one followed by a later section that
        happens to contain a bracket. A quoted annotation may contain anything,
        so nothing inside quotes counts. Once ``maxsplit`` parts are split off,
        the rest is left as it is, unread.
        """
        if '"' not in text and "[" not in text:
            # Nothing can hide a separator, and a stray `]` counts for nothing.
            return text.split(separator, maxsplit or -1)

        parts: list[str] = []
        start = depth = 0
        quoted = False
        special = {'"', "[", "]", separator}

        for i, char in enumerate(text):
            # Most characters are none of these, so they are passed over first.
            if char not in special:
                continue
            if char == '"':
                quoted = not quoted
            elif quoted:
                continue
            elif char == "[":
                depth += 1
            elif char == "]":
                depth = max(depth - 1, 0)
            elif depth == 0:
                parts.append(text[start:i])
                start = i + 1
                if len(parts) == maxsplit:
                    break
        else:
            if quoted:
                raise ValueError(f"Unbalanced quotes in {text}. Check HELM.")
            if depth:
                raise ValueError(f"Unbalanced brackets in {text}. Check HELM.")

        parts.append(text[start:])
        return parts

    @staticmethod
    def _split_helm_sections(helm: str) -> tuple[list[str], list[str], list[str]]:
        """Return the polymers, connections and hydrogen bonds of a HELM string.

        The annotation and version sections that may follow are not used, so
        they are not read either.
        """
        sections = Molecule._split_outside_brackets(helm, "$", maxsplit=3)
        sections += [""] * (3 - len(sections))

        # An empty polymer section still yields one empty polymer, which
        # _process_polymers rejects; the other sections may be empty.
        polymers = Molecule._split_outside_brackets(sections[0], "|")
        connections, hydrogen_bonds = (
            Molecule._split_outside_brackets(section, "|") if section else []
            for section in sections[1:3]
        )
        return polymers, connections, hydrogen_bonds

    @staticmethod
    def _split_sequence_with_brackets(sequence: str) -> list[str]:
        """Split a sequence into individual monomers, respecting brackets.

        Nothing inside a quoted annotation counts, the way nothing inside a
        bracketed monomer does; the split on ``$`` and ``|`` has already made
        sure the quotes are balanced.
        """
        parts: list[str] = []
        start = depth = 0
        quoted = False

        for i, char in enumerate(sequence):
            # Most characters are none of these, so they are passed over first.
            if char not in _SEQUENCE_SPECIAL:
                continue
            if char == '"':
                quoted = not quoted
            elif quoted:
                continue
            elif char in "[(":
                depth += 1
            elif char in "])":
                depth -= 1
                if depth < 0:
                    raise ValueError(
                        f"Unbalanced brackets in sequence {sequence}. Check HELM."
                    )
            elif depth == 0:
                parts.append(sequence[start:i])
                start = i + 1

        if depth:
            raise ValueError(f"Unbalanced brackets in sequence {sequence}. Check HELM.")

        # Appended even when empty: a trailing separator leaves a nameless
        # residue behind, which _process_monomer rejects rather than dropping.
        parts.append(sequence[start:])
        return parts

    @staticmethod
    def _parse_rna_string(sequence: str) -> list[str]:
        """Split an RNA residue such as ``R(A)P`` into its monomers.

        The brackets are left on: _process_monomer strips them and uses their
        presence to tell an inline SMILES monomer from a library symbol.
        """
        parts: list[str] = []
        start = depth = 0

        for i, char in enumerate(sequence):
            if char in "[(":
                depth += 1
            elif char in "])":
                depth -= 1
            if depth == 0:
                parts.append(sequence[start : i + 1])
                start = i + 1

        if start < len(sequence):
            parts.append(sequence[start:])
        return parts

    @staticmethod
    def _sequence_span(chain: str) -> tuple[int, int] | None:
        """Find the braces around a polymer's sequence.

        The closing brace is the first one outside a quoted monomer annotation,
        which may contain braces of its own.
        """
        open_idx = chain.find("{")
        if open_idx < 0:
            return None
        close_idx = chain.find("}", open_idx)
        if close_idx < 0:
            return None
        if '"' not in chain[open_idx:close_idx]:
            return open_idx, close_idx

        quoted = False
        for i in range(open_idx + 1, len(chain)):
            char = chain[i]
            if char == '"':
                quoted = not quoted
            elif char == "}" and not quoted:
                return open_idx, i
        return None

    @staticmethod
    def _strip_annotation(text: str) -> str:
        """Remove the quoted annotation that may close a monomer or connection.

        An annotation is a note for the reader, such as ``"mutation"``, and
        says nothing about the structure.
        """
        if not text.endswith('"'):
            return text
        return text[: text.rfind('"', 0, -1)]

    @staticmethod
    def _expand_residues(residues: list[str], polymer_type: PolymerType) -> list[str]:
        """Drop annotations and write out repeats, such as ``A'3'``.

        A repeat applies to one residue, or to a group of them in parentheses
        such as ``(A.G)'3'``. Connections count residues once repeats are
        written out, the way HELM2 numbers them.
        """
        expanded: list[str] = []
        for residue in residues:
            residue = Molecule._strip_annotation(residue)
            count = 1
            if residue.endswith("'"):
                residue, count = Molecule._parse_repeat(residue)

            group = Molecule._group_content(residue)
            if group is None:
                # Whether R(A)P'2' repeats the nucleotide or only its phosphate
                # is not settled, so only a single monomer is repeated this way.
                if (
                    count > 1
                    and polymer_type == "RNA"
                    and len(Molecule._parse_rna_string(residue)) > 1
                ):
                    raise ValueError(
                        f"Residue {residue}'{count}' repeats more than one monomer; write it as a group, ({residue})'{count}'. Check HELM."
                    )
                expanded.extend([residue] * count)
            else:
                members = Molecule._split_sequence_with_brackets(group)
                expanded.extend(
                    Molecule._expand_residues(members, polymer_type) * count
                )
        return expanded

    @staticmethod
    def _parse_repeat(residue: str) -> tuple[str, int]:
        """Split a residue such as ``A'3'`` into what is repeated and how often."""
        start = residue.rfind("'", 0, -1)
        if start < 0:
            raise ValueError(f"Residue {residue} has a stray quote. Check HELM.")
        if start == 0:
            raise ValueError(f"Residue {residue} repeats nothing. Check HELM.")
        repeated, count = residue[:start], residue[start + 1 : -1]
        if not count.isdecimal():
            if "-" in count:
                raise ValueError(
                    f"Residue {residue} repeats a range of times, which describes more than one molecule."
                )
            raise ValueError(
                f"Residue {residue} has a repeat count {count} that is not a number. Check HELM."
            )
        if int(count) < 1:
            raise ValueError(
                f"Residue {residue} is repeated {count} times; a repeat is at least once. Check HELM."
            )
        return repeated, int(count)

    @staticmethod
    def _group_content(residue: str) -> str | None:
        """The residues inside a group in parentheses, such as ``(A.G)``.

        Parentheses also hold alternatives and ratios, such as ``(A,G)`` and
        ``(A:1+G:2)``, which are not one sequence; those are left alone. So is
        a residue whose parentheses close before its end, such as an RNA
        residue that starts with its base.
        """
        if not (residue.startswith("(") and residue.endswith(")")):
            return None
        depth = 0
        for i, char in enumerate(residue):
            if char in "[(":
                depth += 1
            elif char in "])":
                depth -= 1
                if depth == 0 and i < len(residue) - 1:
                    return None
            elif depth == 1 and char in ",+:":
                return None
        return residue[1:-1]

    @staticmethod
    def _extract_polymer_type(chain_id: str) -> PolymerType:
        """Return the polymer type of a chain ID such as ``PEPTIDE1``."""
        match = Molecule._chain_id_re.fullmatch(chain_id)
        if not match:
            raise ValueError(f"Invalid chain format: {chain_id}")

        polymer_type = match.group(1)
        if polymer_type not in _MONOMER_TYPES:
            raise ValueError(f"Unsupported polymer type: {polymer_type}")

        return polymer_type

    def _process_polymers(self, polymers: list[str]) -> None:
        """Add the monomers of each polymer chain and the bonds along it."""
        for chain in polymers:
            chain = chain.strip()
            span = self._sequence_span(chain)
            if span is None:
                raise ValueError(
                    f"Polymer {chain} is not of the form CHAIN{{sequence}}. Check HELM."
                )
            open_idx, close_idx = span

            trailing = chain[close_idx + 1 :]
            if trailing and not self._annotation_re.fullmatch(trailing):
                raise ValueError(
                    f"Unexpected text {trailing} after the sequence of polymer {chain}. Check HELM."
                )

            chain_id = chain[:open_idx]
            polymer_type = self._extract_polymer_type(chain_id)

            if chain_id in self.chain_offset:
                raise ValueError(f"Duplicate chain ID: {chain_id}")

            sequence = chain[open_idx + 1 : close_idx]
            if not sequence:
                raise ValueError(
                    f"Polymer {chain_id} has an empty sequence. Check HELM."
                )

            residues = self._split_sequence_with_brackets(sequence)
            # Only an annotation, a repeat or a group needs any more work, and a
            # group opens a residue: an RNA base in parentheses never does.
            if (
                '"' in sequence
                or "'" in sequence
                or sequence.startswith("(")
                or ".(" in sequence
            ):
                residues = self._expand_residues(residues, polymer_type)
            self.chain_offset[chain_id] = len(self.monomers)

            if polymer_type == "PEPTIDE":
                self._process_peptide(chain_id, residues)
            elif polymer_type == "RNA":
                self._process_rna(chain_id, residues)
            elif polymer_type == "CHEM":
                if len(residues) != 1:
                    raise ValueError("CHEM polymers must have exactly one residue")
                self._add_monomer(residues[0], chain_id, 0, "chem")
            else:
                assert_never(polymer_type)

    def _process_peptide(self, chain_id: str, residues: list[str]) -> None:
        """Add a peptide chain, bonding each residue's R2 to the next one's R1."""
        for residue_idx, monomer_name in enumerate(residues):
            monomer_idx = self._add_monomer(monomer_name, chain_id, residue_idx, "aa")
            if residue_idx > 0:
                self._add_bond(monomer_idx - 1, 1, monomer_idx, 0, "monomers")

    def _process_rna(self, chain_id: str, residues: list[str]) -> None:
        """Add an RNA chain, where each residue holds several monomers.

        Backbone monomers are bonded R2 to the next one's R1. A monomer in
        parentheses is a branch, such as a base, and hangs off R3 of the
        backbone monomer before it.
        """
        previous = None
        for residue_idx, residue in enumerate(residues):
            parts = self._parse_rna_string(residue)
            if not parts:
                raise ValueError(f"Monomer {residue_idx + 1} has no name. Check HELM.")

            for part in parts:
                is_branch = part.startswith("(") and part.endswith(")")
                monomer_name = part[1:-1] if is_branch else part
                if is_branch and previous is None:
                    raise ValueError(
                        f"Branch monomer ({monomer_name}) starts chain {chain_id} and has nothing to attach to. Check HELM."
                    )

                monomer_idx = self._add_monomer(
                    monomer_name, chain_id, residue_idx, "rna"
                )
                if previous is not None:
                    rgroup = 2 if is_branch else 1
                    self._add_bond(previous, rgroup, monomer_idx, 0, "monomers")
                if not is_branch:
                    previous = monomer_idx

    def _add_monomer(
        self, monomer_name: str, chain_id: str, residue_idx: int, m_type: str
    ) -> int:
        """Append a monomer to the molecule and return its index."""
        self.monomers.append(self._process_monomer(monomer_name, residue_idx, m_type))
        monomer_idx = len(self.monomers) - 1
        self.residue_reps[chain_id].append(monomer_idx)
        return monomer_idx

    def _process_monomer(
        self, monomer_name: str, residue_idx: int, m_type: str
    ) -> MonomerData:
        """Look a monomer up in the library, or build it from inline SMILES."""
        bracketed = monomer_name.startswith("[") and monomer_name.endswith("]")
        if bracketed:
            monomer_name = monomer_name[1:-1]
        if not monomer_name:
            raise ValueError(f"Monomer {residue_idx + 1} has no name. Check HELM.")

        # An ambiguous monomer (a,[b]) is built as its last alternative.
        ambiguous = self._ambiguous_re.fullmatch(monomer_name)
        if ambiguous:
            self.has_ambiguous_monomers = True
            return self._process_monomer(f"[{ambiguous.group(1)}]", residue_idx, m_type)

        monomer = self.monomer_df.get(m_type, {}).get(monomer_name)
        if monomer is None:
            if not bracketed:
                # HELM writes inline SMILES in square brackets, so a bare name
                # is a library symbol. Falling back to SMILES here would read a
                # typo such as PEPTIDE1{B} as a boron atom and build it without
                # complaining.
                raise ValueError(
                    f"Monomer {monomer_name} is not in the {m_type} monomer library. Inline SMILES has to be written in square brackets. Check HELM."
                )
            # Deliberately not written back into monomer_df: that dictionary is
            # shared by every Molecule through the cached monomer library, so
            # writing here would grow it with every inline monomer ever parsed.
            monomer = _create_missing_monomer(monomer_name, m_type)

        # The cap groups are copied because bonding clears the ones it uses.
        return {**monomer, "m_Rgroups": monomer["m_Rgroups"][:]}

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
        return rgroup - 1

    @staticmethod
    def _parse_connection(connection_str: str) -> tuple[str, int, int, str, int, int]:
        """Parse a connection into its two chains, 0-based residues and R-groups.

        A connection that cannot be parsed is an error rather than a warning:
        skipping it would return a molecule that is quietly missing a bond.
        """
        parts = connection_str.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid connection format: {connection_str}. Check HELM."
            )

        chain_id1, chain_id2, bond_spec = parts

        bond_parts = Molecule._bond_spec_re.split(bond_spec)
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

    def _process_connections(self, connections: list[str]) -> None:
        """Add the bonds the connection section declares.

        HELM2 writes hydrogen bonds here too, as ``pair`` in place of both
        R-groups; they are recorded rather than bonded.
        """
        for connection_str in map(self._strip_annotation, connections):
            if self._is_hydrogen_bond(connection_str):
                self._add_hydrogen_bond(connection_str)
                continue
            chain_id1, residue1, rgroup1, chain_id2, residue2, rgroup2 = (
                self._parse_connection(connection_str)
            )
            monomer_idx1 = self._resolve_residue(chain_id1, residue1, "connections")
            monomer_idx2 = self._resolve_residue(chain_id2, residue2, "connections")
            self._add_bond(monomer_idx1, rgroup1, monomer_idx2, rgroup2, "connections")

    @staticmethod
    def _is_hydrogen_bond(connection_str: str) -> bool:
        """Does a connection name ``pair`` in place of an R-group?"""
        bond_spec = connection_str.rsplit(",", 1)[-1]
        bond_parts = Molecule._bond_spec_re.split(bond_spec)
        return len(bond_parts) == 4 and "pair" in (bond_parts[1], bond_parts[3])

    def _process_hydrogen_bonds(self, connections: list[str]) -> None:
        """Record the hydrogen bonds of HELM1's own section for them.

        HELM2 gives this section to polymer groups instead, such as
        ``G1(PEPTIDE1+CHEM1:2.5)``. A group gathers polymers into a mixture, a
        ratio or a set of alternatives, none of which is one molecule.
        """
        for connection_str in map(self._strip_annotation, connections):
            if self._polymer_group_re.fullmatch(connection_str):
                raise ValueError(
                    f"Polymer group {connection_str} describes a mixture or a choice of polymers rather than one molecule, which helmkit cannot build."
                )
            self._add_hydrogen_bond(connection_str)

    def _add_hydrogen_bond(self, connection_str: str) -> None:
        """Record a hydrogen bond; it adds no bond to the molecule."""
        parts = connection_str.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid hydrogen bond format: {connection_str}. Check HELM."
            )
        chain_id1, chain_id2, bond_spec = parts

        bond_parts = self._bond_spec_re.split(bond_spec)
        if len(bond_parts) != 4:
            raise ValueError(f"Invalid hydrogen bond format: {bond_spec}. Check HELM.")
        # Anything else would be a covalent bond written where only hydrogen
        # bonds go, and recording it as one would quietly leave the bond out.
        if bond_parts[1] != "pair" or bond_parts[3] != "pair":
            raise ValueError(
                f"Hydrogen bond {bond_spec} has to name pair at both ends. Check HELM."
            )

        residue1 = self._parse_residue_number(bond_parts[0], bond_spec)
        residue2 = self._parse_residue_number(bond_parts[2], bond_spec)
        self._resolve_residue(chain_id1, residue1, "hydrogen bonds")
        self._resolve_residue(chain_id2, residue2, "hydrogen bonds")
        self.hydrogen_bonds.append([chain_id1, residue1, chain_id2, residue2])

    @staticmethod
    def _attachment_point(
        monomer: MonomerData, rgroup: int, position: int, context: str
    ) -> int:
        """Look up the atom a monomer's 0-based R-group bonds through.

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

    def _add_bond(
        self,
        monomer_idx1: int,
        rgroup1: int,
        monomer_idx2: int,
        rgroup2: int,
        context: str,
    ) -> None:
        """Bond two monomers through their 0-based R-groups."""
        atom1 = self._attachment_point(
            self.monomers[monomer_idx1], rgroup1, monomer_idx1 + 1, context
        )
        atom2 = self._attachment_point(
            self.monomers[monomer_idx2], rgroup2, monomer_idx2 + 1, context
        )
        self.bondlist.append([monomer_idx1, atom1, monomer_idx2, atom2])
        self.bond_rgroups.append((rgroup1, rgroup2))
        self._mark_used_rgroup(monomer_idx1, rgroup1)
        self._mark_used_rgroup(monomer_idx2, rgroup2)

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

    # Building

    def _build_molecule(self) -> Chem.Mol:
        """Build the RDKit molecule from parsed monomer and bond data."""
        mol = Chem.RWMol(self.monomers[0]["m_romol"])
        for monomer in self.monomers[1:]:
            mol.InsertMol(monomer["m_romol"])

        sizes = (monomer["m_romol"].GetNumAtoms() for monomer in self.monomers)
        self.offset = [0, *itertools.accumulate(sizes)]

        # The hydrogens each atom capped with hydrogen should end up with.
        hydrogens: dict[int, int] = {}
        for monomer_idx, offset in enumerate(self.offset[:-1]):
            self._replace_rgroups(mol, offset, monomer_idx, hydrogens)
        self._add_bonds(mol)
        return self._remove_rgroup_atoms(mol, hydrogens)

    def _replace_rgroups(
        self,
        mol: Chem.RWMol,
        atom_offset: int,
        monomer_idx: int,
        hydrogens: dict[int, int],
    ) -> None:
        """Cap every R-group of a monomer that no bond has claimed.

        A hydrogen cap needs no atom of its own: the dummy is deleted and RDKit
        fills the free valence with an implicit hydrogen. Not every atom gets
        one, so the hydrogens the atom should end up with are noted in
        ``hydrogens`` for _remove_rgroup_atoms to make good.
        """
        monomer = self.monomers[monomer_idx]
        for rgroup, (cap, atom_idx, attachment) in enumerate(
            zip(
                monomer["m_Rgroups"],
                monomer["m_RgroupIdx"],
                monomer["m_attachmentPointIdx"],
            )
        ):
            if atom_idx is None or attachment is None:
                if cap is not None:
                    raise ValueError(
                        f"R-group {rgroup + 1} of monomer {monomer['m_abbr']} has the cap group {cap} but no atom index."
                    )
                continue
            if (monomer_idx, rgroup) in self.used_rgroups:
                continue
            if cap is None or cap == "H":
                atom_idx = atom_offset + attachment
                if atom_idx not in hydrogens:
                    hydrogens[atom_idx] = mol.GetAtomWithIdx(atom_idx).GetTotalNumHs()
                hydrogens[atom_idx] += 1
            else:
                self._replace_rgroup(mol, atom_offset + atom_idx, cap)

    @staticmethod
    def _replace_rgroup(mol: Chem.RWMol, atom_idx: int, cap: str) -> None:
        """Replace an unused R-group with the heavy atom of its cap group."""
        # Leaving the cap group off would delete the dummy atom along with the
        # used R-groups and hand back a molecule that is quietly missing an atom.
        atomic_number = _cap_group_atomic_number(cap)
        if atomic_number is None:
            raise ValueError(f"Unsupported R-group cap group {cap}. Check monomers.")

        try:
            mol.ReplaceAtom(atom_idx, Chem.Atom(atomic_number))
        except (RuntimeError, OverflowError) as e:
            raise ValueError(
                f"Failed to replace R-group with {cap}: {e}. Check monomers."
            ) from e

    def _bond_atoms(self) -> Iterator[tuple[int, int, int, int]]:
        """Yield each bond in the bond list as (monomer1, atom1, monomer2, atom2).

        The atom indices are absolute, in the molecule as the offsets describe it.
        """
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self.bondlist:
            yield (
                monomer1_idx,
                self.offset[monomer1_idx] + atom1_idx,
                monomer2_idx,
                self.offset[monomer2_idx] + atom2_idx,
            )

    def _add_bonds(self, mol: Chem.RWMol) -> None:
        """Add bonds between monomers based on bond list."""
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self._bond_atoms():
            # RDKit answers both of these with a C++ pre-condition violation,
            # which tells a caller nothing about which connection is at fault.
            if atom1_idx == atom2_idx:
                raise ValueError(
                    f"Monomer {monomer1_idx + 1} is bonded to itself through one atom. Check connections."
                )
            if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
                raise ValueError(
                    f"Duplicate bond between monomer {monomer1_idx + 1} and monomer {monomer2_idx + 1}. Check connections."
                )

            mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)

    def _rgroup_atom(self, monomer_idx: int, rgroup: int) -> int:
        """The absolute index of a monomer's 0-based R-group atom."""
        atom_idx = self.monomers[monomer_idx]["m_RgroupIdx"][rgroup]
        # Every bond is made through an R-group with an attachment point, which
        # only an R-group with an atom has.
        assert atom_idx is not None
        return self.offset[monomer_idx] + atom_idx

    def _move_stereo_references(self, mol: Chem.RWMol, deleted: set[int]) -> None:
        """Move double bond stereo references off the atoms about to be deleted.

        RDKit drops a reference that no longer exists, and a double bond with no
        references loses the geometry the monomer declared. The atom bonded
        through an R-group stands where that R-group stood, so it takes the
        reference over unchanged. An R-group no bond used leaves only an
        implicit hydrogen in its place, so the other surviving neighbour takes
        over instead: it is on the opposite side and flips the parity.
        """
        # A stereo reference is a neighbour of the double bond, so only a double
        # bond next to a deleted atom can lose one. Finding those by
        # substructure search is far quicker than walking every bond.
        matches = mol.GetSubstructMatches(
            _DUMMY_BY_DOUBLE_BOND, uniquify=False, maxMatches=_ALL_MATCHES
        )
        candidates = sorted({
            mol.GetBondBetweenAtoms(end, other_end).GetIdx()
            for _, end, other_end in matches
        })
        if not candidates:
            return

        # Each R-group atom a bond was made through, and the atom bonded in its
        # place. Which R-group matters, not just which atom it was on: an atom
        # can carry two, and the atom bonded through one is not where the other
        # stood.
        replaced_by: dict[int, int] = {}
        for (monomer1, first, monomer2, second), (rgroup1, rgroup2) in zip(
            self._bond_atoms(), self.bond_rgroups
        ):
            replaced_by[self._rgroup_atom(monomer1, rgroup1)] = second
            replaced_by[self._rgroup_atom(monomer2, rgroup2)] = first

        for bond in map(mol.GetBondWithIdx, candidates):
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
                if reference in replaced_by:
                    moved.append(replaced_by[reference])
                    continue
                end = next(
                    (e for e in ends if mol.GetBondBetweenAtoms(e, reference)), None
                )
                if end is None:
                    break
                other_end = ends[1] if end == ends[0] else ends[0]
                others = [
                    neighbour.GetIdx()
                    for neighbour in mol.GetAtomWithIdx(end).GetNeighbors()
                    if neighbour.GetIdx() not in deleted
                    and neighbour.GetIdx() != other_end
                ]
                if len(others) != 1:
                    break
                flip = not flip
                moved.append(others[0])

            if len(moved) != 2:
                # Nothing dependable to point at, so drop the geometry rather
                # than state one that might be the wrong way round.
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                continue

            bond.SetStereoAtoms(*moved)
            if flip:
                bond.SetStereo(_FLIPPED_STEREO[bond.GetStereo()])

    def _remove_rgroup_atoms(
        self, mol: Chem.RWMol, hydrogens: dict[int, int]
    ) -> Chem.Mol:
        """Delete the R-group atoms left over and renumber the bond list to match.

        ``hydrogens`` gives the hydrogens each atom capped with hydrogen should
        end up with, by its index before the deletion.
        """
        rgroup_atoms = _dummy_atoms(mol)
        self._move_stereo_references(mol, set(rgroup_atoms))
        _delete_atoms(mol, rgroup_atoms)

        def shifted(idx: int) -> int:
            """Where an atom that was not deleted ends up."""
            return idx - bisect.bisect_left(rgroup_atoms, idx)

        # RDKit gives no implicit hydrogens to an atom whose hydrogen count is
        # fixed, such as any bracket atom in SMILES, nor to an aromatic
        # nitrogen, which it cannot tell from one that needs none. Either is
        # left a hydrogen short, a radical or a ring that cannot be kekulized,
        # so only the hydrogens RDKit did not add are made explicit.
        for atom_idx, expected in hydrogens.items():
            atom = mol.GetAtomWithIdx(shifted(atom_idx))
            missing = expected - atom.GetTotalNumHs()
            if missing > 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + missing)
                atom.UpdatePropertyCache(strict=False)

        for bond in self.bondlist:
            monomer1_idx, atom1_idx, monomer2_idx, atom2_idx = bond
            offset1, offset2 = self.offset[monomer1_idx], self.offset[monomer2_idx]
            bond[1] = shifted(offset1 + atom1_idx) - shifted(offset1)
            bond[3] = shifted(offset2 + atom2_idx) - shifted(offset2)
        self.offset = [shifted(offset) for offset in self.offset]

        result = mol.GetMol()

        # Ring membership is not worked out by any of the above, and RDKit
        # answers a ring query on a molecule without it by raising, which takes
        # out ring descriptors and every SMARTS match that mentions a ring.
        # SetDoubleBondNeighborDirections works out the symmetrized SSSR itself
        # when it is missing, so asking for that here costs nothing extra.
        Chem.GetSymmSSSR(result)

        # Stereo is carried on the double bond, but writing SMILES needs the
        # direction of the single bonds around it, which nothing above sets. A
        # molecule would report its geometry through InChI and lose it through
        # SMILES.
        Chem.SetDoubleBondNeighborDirections(result)
        return result

    # Results

    @property
    def bond_indices(self) -> list[int]:
        """The index in `mol` of each bond in the bond list."""
        indices = []
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self._bond_atoms():
            bond = self.mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            if bond is None:
                raise ValueError(
                    f"The bond between monomer {monomer1_idx + 1} and monomer {monomer2_idx + 1} is not present in the molecule."
                )
            indices.append(bond.GetIdx())
        return indices

    @property
    def monomer_indices(self) -> list[int]:
        """The index of the monomer each atom of `mol` comes from."""
        return [
            bisect.bisect_right(self.offset, i) - 1
            for i in range(self.mol.GetNumAtoms())
        ]


_monomer_df: MonomerLibrary | None = None


def _init_pool(monomer_df: MonomerLibrary) -> None:
    global _monomer_df
    _monomer_df = monomer_df


def _load_helm(helm: str) -> Molecule:
    molecule = Molecule(helm, _monomer_df)
    # The library is only needed while parsing and the parent already has it,
    # so it is not pickled back with every chunk of results.
    del molecule.monomer_df
    return molecule


def load_in_parallel(
    helms: Iterable[str],
    monomer_df: MonomerLibrary | None = None,
    chunksize: int | None = 256,
) -> list[Molecule]:
    """Build a Molecule for each HELM string using a pool of worker processes."""
    if monomer_df is None:
        monomer_df = load_monomer_library()
    with multiprocessing.Pool(initializer=_init_pool, initargs=(monomer_df,)) as pool:
        molecules = pool.map(_load_helm, helms, chunksize=chunksize)
    for molecule in molecules:
        molecule.monomer_df = monomer_df
    return molecules
