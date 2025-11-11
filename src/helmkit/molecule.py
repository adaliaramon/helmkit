import bisect
import multiprocessing
import re
import warnings
from collections import defaultdict
from functools import lru_cache
from importlib.resources import files
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypedDict
from typing import TypeVar
from typing import Union

from rdkit import Chem
from rdkit import rdBase


class SequenceConstants:
    max_rgroups: int = 4


def get_molecule_property(
    molecule: Chem.Mol, property_name: str, default: Optional[str] = None
) -> Optional[str]:
    return (
        molecule.GetProp(property_name) if molecule.HasProp(property_name) else default
    )


T = TypeVar("T")


def parse_comma_separated_property(
    molecule: Chem.Mol,
    property_name: str,
    convert_func: Optional[Callable[[str], T]] = None,
) -> Sequence[Optional[Union[str, T]]]:
    property_value = get_molecule_property(molecule, property_name)
    if not property_value:
        return []

    values = property_value.split(",")
    if convert_func:
        values = [convert_func(v) if v != "None" else None for v in values]
    else:
        values = [None if v == "None" else v for v in values]

    return values


def infer_attachment_points(
    molecule: Chem.Mol, rgroup_indices: Sequence[Optional[int]]
) -> List[int]:
    """Infer attachment points by finding atoms bonded to R-group atoms."""
    attachment_points = []

    for r_idx in rgroup_indices:
        if r_idx is None:
            attachment_points.append(None)
            continue

        atom = molecule.GetAtomWithIdx(r_idx)

        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(r_idx)
            attachment_points.append(other_idx)
            break
        else:
            attachment_points.append(None)
            warnings.warn(
                f"R-group atom {r_idx} has no bonds to determine attachment point"
            )

    return attachment_points


class MonomerData(TypedDict):
    m_romol: Chem.Mol
    m_Rgroups: List[str]
    m_RgroupIdx: List[int]
    m_attachmentPointIdx: List[int]
    m_type: str
    m_subtype: str
    m_abbr: str


MonomerLibrary = Dict[str, Dict[str, MonomerData]]


@lru_cache
def load_monomer_library(library_path: Optional[str] = None) -> MonomerLibrary:
    """Load and prepare monomer data from SDF file."""
    if library_path is None:
        library_path = str(files("helmkit.data") / "monomers.sdf")
    monomers_dict: MonomerLibrary = defaultdict(dict)
    supplier = Chem.SDMolSupplier(library_path, removeHs=False)

    for mol in supplier:
        if mol is None:
            continue

        symbol = get_molecule_property(mol, "symbol")
        if not symbol:
            continue

        m_type = get_molecule_property(mol, "m_type", "")

        rgroups = parse_comma_separated_property(mol, "m_Rgroups")
        rgroup_idx = parse_comma_separated_property(mol, "m_RgroupIdx", int)
        attachment_point_idx = infer_attachment_points(mol, rgroup_idx)

        monomers_dict[m_type][symbol] = {
            "m_romol": mol,
            "m_Rgroups": rgroups,
            "m_RgroupIdx": rgroup_idx,
            "m_attachmentPointIdx": attachment_point_idx,
            "m_type": m_type,
            "m_subtype": get_molecule_property(mol, "m_subtype", ""),
            "m_abbr": get_molecule_property(mol, "m_abbr", ""),
        }

    return monomers_dict


def _create_missing_monomer(monomer_name: str, m_type: str = "aa") -> MonomerData:
    mol = Chem.MolFromSmiles(monomer_name, sanitize=False)
    if mol is None:
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

    r_group_map = {}
    main_atoms = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        label = atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else ""

        if label.startswith("_R"):
            try:
                r_num = int(label[2:])
                atom.SetProp("dummyLabel", f"R{r_num}")
                atom.SetIntProp("_MolFileRLabel", r_num)
                atom.SetProp("molFileValue", "*")
                r_group_map[r_num] = idx
            except ValueError:
                continue
        else:
            main_atoms.append(idx)

    sorted_r = sorted(r_group_map.items())
    r_group_idx = [idx for _, idx in sorted_r]
    mol = Chem.RenumberAtoms(mol, main_atoms + r_group_idx)

    rgroup_idx_full: List[Optional[int]] = [None] * SequenceConstants.max_rgroups
    for i, (r_num, _) in enumerate(sorted_r):
        if 1 <= r_num <= SequenceConstants.max_rgroups:
            rgroup_idx_full[r_num - 1] = len(main_atoms) + i

    attachment_points = infer_attachment_points(mol, rgroup_idx_full)
    rgroup_vals = [None] * SequenceConstants.max_rgroups

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
        aldehyde = Chem.MolFromSmarts("[CX3H1]=O")
        matches = mol.GetSubstructMatches(aldehyde)
        if len(matches) == 0:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OH]"))
        if len(matches) == 1:
            attachment_id, *_ = matches[0]

            mol = Chem.RWMol(mol)
            new_idx = mol.AddAtom(Chem.Atom(0))
            mol.AddBond(attachment_id, new_idx, Chem.BondType.SINGLE)
            rgroup_idx_full[1] = new_idx
            attachment_points[1] = attachment_id

    mol.SetProp("m_name", monomer_name)

    mol.SetProp("symbol", monomer_name)
    mol.SetProp("m_abbr", monomer_name)
    mol.SetProp("m_type", m_type)
    mol.SetProp("m_subtype", "non-natural" if m_type == "aa" else "")
    mol.SetProp("m_RgroupIdx", ",".join(map(str, rgroup_idx_full)))
    mol.SetProp("m_Rgroups", ",".join(map(str, rgroup_vals)))
    mol.SetProp("m_attachmentPointIdx", ",".join(map(str, attachment_points)))
    mol.SetProp("natAnalog", "")

    monomer = {
        "m_romol": mol,
        "m_Rgroups": rgroup_vals,
        "m_RgroupIdx": rgroup_idx_full,
        "m_attachmentPointIdx": attachment_points,
        "m_type": m_type,
        "m_subtype": "non-natural" if m_type == "aa" else "",
        "m_abbr": monomer_name,
    }
    return monomer


class Molecule:
    """Single class for HELM to RDKit Mol conversion."""

    _bracket_re = re.compile(r"{(.*?)}")
    _pipe_outside_brackets = re.compile(r"\|(?![^\[]*\])")
    _dollar_outside_brackets = re.compile(r"\$(?![^\[]*\])")

    def __init__(self, helm: str, monomer_df: Optional[MonomerLibrary] = None):
        """Initialize a Molecule object from a HELM string."""
        self.mol = None
        self.offset = []
        self.bondlist = []
        self.monomers = []
        self.chain_offset = {}
        self.residue_reps = defaultdict(list)
        self.has_ambiguous_monomers = False
        self.hydrogen_bonds = []

        if monomer_df is None:
            self.monomer_df = load_monomer_library()
        else:
            self.monomer_df = monomer_df

        self._parse_helm_string(helm)
        self._build_molecule()

        if not isinstance(self.mol, Chem.rdchem.Mol):
            raise RuntimeError("Failed to initialize RDKit Mol object")

    def _parse_helm_string(self, helm: str) -> None:
        """Parse a HELM string into molecular components."""
        helm_parts = self._split_helm_sections(helm)

        if len(helm_parts) < 5:
            warnings.warn(f"Problem with HELM string - not enough sections: {helm}")
            return

        polymer_sections, connection_sections, hydrogen_bonds_sections = (
            helm_parts[0],
            helm_parts[1],
            helm_parts[2],
        )

        if not polymer_sections:
            warnings.warn(f"No simple polymers in HELM string {helm}")
            return

        self._process_polymers(polymer_sections)
        self._process_connections(connection_sections)
        self._process_hydrogen_bonds(hydrogen_bonds_sections)

    def _split_helm_sections(self, helm: str) -> List[Union[str, List[str]]]:
        """Split a HELM string into its components."""
        parts = self._dollar_outside_brackets.split(helm, 4)
        parts.extend([""] * (5 - len(parts)))

        parts[0] = (
            self._pipe_outside_brackets.split(parts[0])
            if "|" in parts[0]
            else [parts[0]]
        )

        if parts[1]:
            parts[1] = parts[1].split("|") if "|" in parts[1] else [parts[1]]
        else:
            parts[1] = []

        if parts[2]:
            parts[2] = parts[2].split("|") if "|" in parts[2] else [parts[2]]
        else:
            parts[2] = []

        return parts

    @staticmethod
    def _split_sequence_with_brackets(sequence: str) -> List[str]:
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
                current += char
            elif char == "." and bracket_depth == 0:
                result.append(current)
                current = ""
            else:
                current += char

        if current:
            result.append(current)

        return result

    def _extract_chain_id(
        self, chain_str: str
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        """Extract chain ID and validate chain type."""
        match = re.match(r"([A-Z]+)(\d+)", chain_str)
        if not match:
            warnings.warn(f"Invalid chain format: {chain_str}")
            return None, False, None

        polymer_type = match.group(1)
        if polymer_type not in ("PEPTIDE", "RNA", "CHEM"):
            warnings.warn(f"Unsupported polymer type: {polymer_type}")
            return None, False, None

        try:
            chain_id = chain_str
            return chain_id, True, polymer_type
        except ValueError:
            warnings.warn(f"Invalid chain ID in: {chain_str}")
            return None, False, None

    def _process_monomer(
        self, monomer_name: str, chain_id: str, residue_idx: int, polymer_type: str
    ) -> Optional[MonomerData]:
        """Process a single monomer."""
        monomer_name = (
            monomer_name[1:-1]
            if monomer_name.startswith("[") and monomer_name.endswith("]")
            else monomer_name
        )
        if monomer_name == "":
            raise ValueError(f"Monomer {residue_idx + 1} has no name. Check HELM.")

        # Check for (a,[b]) pattern
        match = re.fullmatch(r"\([^,]+,\[([^\]]+)\]\)", monomer_name)
        if match:
            # Extract the 'b' from (a,[b]) and recurse
            self.has_ambiguous_monomers = True
            return self._process_monomer(
                match.group(1), chain_id, residue_idx, polymer_type
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
        else:
            try:
                monomer_info = _create_missing_monomer(monomer_name, m_type)
                if m_type not in self.monomer_df:
                    self.monomer_df[m_type] = {}
                self.monomer_df[m_type][monomer_name] = monomer_info
            except ValueError as e:
                warnings.warn(str(e))
                return None

        return {
            "m_name": monomer_name,
            "m_chainID": chain_id,
            "m_resID": residue_idx,
            "m_romol": monomer_info["m_romol"],
            "m_Rgroups": monomer_info["m_Rgroups"][:],
            "m_RgroupIdx": monomer_info["m_RgroupIdx"],
            "m_attachmentPointIdx": monomer_info["m_attachmentPointIdx"],
            "m_type": monomer_info["m_type"],
            "m_subtype": monomer_info["m_subtype"],
            "m_abbr": monomer_info["m_abbr"],
        }

    @staticmethod
    def _parse_rna_string(sequence: str) -> List[str]:
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

        result = [
            r[1:-1] if r.startswith("[") and r.endswith("]") else r for r in result
        ]
        return result

    def _process_polymers(self, polymers: List[str]) -> None:
        """Process polymer chains from HELM, creating backbone bonds on the fly."""
        monomer_idx = 0

        for chain in polymers:
            chain = chain.strip()
            match = self._bracket_re.search(chain)
            if not match:
                warnings.warn(f"No sequence in polymer: {chain}")
                continue

            id_chain = chain[: match.start()]
            chain_id, valid, polymer_type = self._extract_chain_id(id_chain)
            if not valid:
                continue

            if chain_id in self.chain_offset:
                raise ValueError(f"Duplicate chain ID: {chain_id}")

            sequence = match.group(1)
            if not sequence:
                warnings.warn(f"Empty polymer: {chain}")
                continue

            residues = self._split_sequence_with_brackets(sequence)
            self.chain_offset[chain_id] = monomer_idx

            if polymer_type == "PEPTIDE":
                for residue_idx, monomer_name in enumerate(residues):
                    monomer = self._process_monomer(
                        monomer_name, chain_id, residue_idx, polymer_type
                    )
                    if not monomer:
                        continue

                    self.monomers.append(monomer)
                    self.residue_reps[chain_id].append(monomer_idx)

                    if residue_idx > 0:
                        monomer1 = self.monomers[monomer_idx - 1]
                        monomer2 = monomer

                        attachment_point1 = monomer1["m_attachmentPointIdx"][1]
                        if attachment_point1 is None:
                            raise ValueError(
                                f"R-group 2 is not present in monomer {monomer_idx} ({monomer1['m_name']}). Check monomers."
                            )
                        attachment_point2 = monomer2["m_attachmentPointIdx"][0]
                        if attachment_point2 is None:
                            raise ValueError(
                                f"R-group 1 is not present in monomer {monomer_idx + 1} ({monomer2['m_name']}). Check monomers."
                            )

                        self.bondlist.append(
                            [
                                monomer_idx - 1,
                                attachment_point1,
                                monomer_idx,
                                attachment_point2,
                            ]
                        )
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
                        monomer_name = (
                            monomer_name[1:-1]
                            if monomer_name.startswith("[")
                            and monomer_name.endswith("]")
                            else monomer_name
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
                            try:
                                attachment_point1 = monomer1["m_attachmentPointIdx"][
                                    r_index
                                ]
                            except IndexError:
                                attachment_point1 = None
                            if attachment_point1 is None:
                                raise ValueError(
                                    f"R-group {r_index + 1} is not present in monomer {prev_monomer} ({monomer1['m_name']}). Check monomers."
                                )
                            attachment_point2 = monomer2["m_attachmentPointIdx"][0]
                            if attachment_point2 is None:
                                raise ValueError(
                                    f"R-group 1 is not present in monomer {monomer_idx} ({monomer2['m_name']}). Check monomers."
                                )

                            self.bondlist.append(
                                [
                                    prev_monomer,
                                    attachment_point1,
                                    monomer_idx,
                                    attachment_point2,
                                ]
                            )
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
                monomer_name = (
                    monomer_name[1:-1]
                    if monomer_name.startswith("[") and monomer_name.endswith("]")
                    else monomer_name
                )
                monomer = self._process_monomer(
                    monomer_name, chain_id, residue_idx, polymer_type
                )
                self.monomers.append(monomer)
                self.residue_reps[chain_id].append(monomer_idx)
                monomer_idx += 1

    def _parse_connection(
        self, connection_str: str
    ) -> Optional[Tuple[str, int, int, str, int, int]]:
        """Parse a single connection string."""
        parts = connection_str.split(",")
        if len(parts) != 3:
            warnings.warn(f"Invalid connection format: {connection_str}")
            return None

        chain_id1, chain_id2, bond_spec = parts

        try:
            bond_parts = re.split(r"[-:]", bond_spec)
            if len(bond_parts) != 4:
                warnings.warn(f"Invalid bond format: {bond_spec}")
                return None

            residue1, rgroup1, residue2, rgroup2 = bond_parts

            residue1 = int(residue1) - 1
            residue2 = int(residue2) - 1
            rgroup1 = int(rgroup1.replace("R", ""))
            rgroup2 = int(rgroup2.replace("R", ""))

            return chain_id1, residue1, rgroup1, chain_id2, residue2, rgroup2
        except (ValueError, IndexError) as e:
            warnings.warn(f"Error parsing connection {connection_str}: {e}")
            return None

    def _process_connections(self, connections: List[str]) -> None:
        """Process connections between chains."""
        if not connections:
            return

        for connection_str in connections:
            parsed = self._parse_connection(connection_str)
            if not parsed:
                continue

            chain_id1, residue1, rgroup1, chain_id2, residue2, rgroup2 = parsed
            rgroup1 -= 1
            rgroup2 -= 1

            monomer_idx1 = self.residue_reps[chain_id1][residue1]
            monomer_idx2 = self.residue_reps[chain_id2][residue2]

            monomer1 = self.monomers[monomer_idx1]
            monomer2 = self.monomers[monomer_idx2]

            attachment_idx1 = monomer1["m_attachmentPointIdx"][rgroup1]
            if attachment_idx1 is None:
                raise ValueError(
                    f"R-group {rgroup1} is not present in monomer {monomer_idx1 + 1} ({monomer1['m_name']}). Check connections."
                )
            attachment_idx2 = monomer2["m_attachmentPointIdx"][rgroup2]
            if attachment_idx2 is None:
                raise ValueError(
                    f"R-group {rgroup2} is not present in monomer {monomer_idx2 + 1} ({monomer2['m_name']}). Check connections."
                )

            self.bondlist.append(
                [monomer_idx1, attachment_idx1, monomer_idx2, attachment_idx2]
            )

            self._mark_used_rgroup(monomer_idx1, rgroup1)
            self._mark_used_rgroup(monomer_idx2, rgroup2)

    def _process_hydrogen_bonds(self, connections: List[str]) -> None:
        """Process hydrogen bonds."""
        if not connections:
            return

        for connection_str in connections:
            parts = connection_str.split(",")
            if len(parts) != 3:
                warnings.warn(f"Invalid hydrogen bond format: {connection_str}")
                continue
            chain_id1, chain_id2, bond_spec = parts

            bond_parts = re.split(r"[-:]", bond_spec)
            if len(bond_parts) != 4:
                warnings.warn(f"Invalid hydrogen bond format: {bond_spec}")
                continue

            residue1, _, residue2, _ = bond_parts
            residue1 = int(residue1) - 1
            residue2 = int(residue2) - 1
            self.hydrogen_bonds.append([chain_id1, residue1, chain_id2, residue2])

    def _mark_used_rgroup(self, monomer_idx: int, rgroup: int) -> None:
        """Mark an R-group as used based on its attachment point index."""
        monomer = self.monomers[monomer_idx]
        monomer["m_Rgroups"][rgroup] = None

    def _build_molecule(self) -> None:
        """Build the RDKit molecule from parsed monomer and bond data."""
        if not self.monomers:
            self.mol = Chem.RWMol()
            return

        monomer = self.monomers[0]
        self.mol = Chem.RWMol(monomer["m_romol"])

        rgroups = monomer["m_Rgroups"]
        rgroup_idx = monomer["m_RgroupIdx"]
        for i in range(min(len(rgroups), SequenceConstants.max_rgroups)):
            if rgroups[i] is not None:
                self._replace_rgroup(self.mol, 0, rgroup_idx[i], rgroups[i])

        current_offset = self.mol.GetNumAtoms()
        self.offset = [0, current_offset]

        for monomer in self.monomers[1:]:
            self.mol.InsertMol(monomer["m_romol"])

            rgroups = monomer["m_Rgroups"]
            rgroup_idx = monomer["m_RgroupIdx"]
            for i in range(min(len(rgroups), SequenceConstants.max_rgroups)):
                if rgroups[i] is not None:
                    self._replace_rgroup(
                        self.mol, current_offset, rgroup_idx[i], rgroups[i]
                    )

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

            self.mol.AddBond(
                absolute_atom1_idx, absolute_atom2_idx, Chem.BondType.SINGLE
            )

    def _replace_rgroup(
        self, rdkit_mol: Chem.RWMol, atom_offset: int, atom_idx: int, atom_type: str
    ) -> None:
        """Replace an R-group with the appropriate atom type."""
        absolute_idx = atom_offset + atom_idx

        if atom_type == "OH":
            try:
                oxygen_atom = Chem.Atom(8)  # Oxygen
                rdkit_mol.ReplaceAtom(absolute_idx, oxygen_atom)
            except Exception as e:
                warnings.warn(f"Failed to replace R-group with OH: {e}")
        elif atom_type != "H":
            warnings.warn(f"Unrecognized R-group type: {atom_type}")

    def _sanitize(self) -> None:
        """Clean up the molecule by removing dummy atoms."""
        pattern = Chem.MolFromSmarts("[#0]")
        matches = self.mol.GetSubstructMatches(pattern)
        atoms_to_delete = sorted({idx for match in matches for idx in match})
        self.mol = Chem.DeleteSubstructs(self.mol, pattern)

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

    @property
    def bond_indices(self) -> List[int]:
        return [
            self.mol.GetBondBetweenAtoms(
                self.offset[monomer1_idx] + atom1_idx,
                self.offset[monomer2_idx] + atom2_idx,
            ).GetIdx()
            for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self.bondlist
        ]

    @property
    def monomer_indices(self) -> List[int]:
        return [
            bisect.bisect_right(self.offset, i) - 1
            for i in range(self.mol.GetNumAtoms())
        ]


def _init_pool(monomer_df: MonomerLibrary):
    global _monomer_df
    _monomer_df = monomer_df


def _load_peptide(helm: str) -> Molecule:
    return Molecule(helm, _monomer_df)


def load_peptides_in_parallel(
    helms: List[str],
    monomer_df: Optional[MonomerLibrary] = None,
    chunksize: Optional[int] = 256,
) -> List[Molecule]:
    if monomer_df is None:
        monomer_df = load_monomer_library()
    with multiprocessing.Pool(initializer=_init_pool, initargs=(monomer_df,)) as pool:
        return pool.map(_load_peptide, helms, chunksize=chunksize)
