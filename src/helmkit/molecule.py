import multiprocessing
import re
import warnings
from functools import lru_cache
from importlib.resources import files
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from rdkit import Chem


class SequenceConstants:
    max_rgroups = 4


def get_molecule_property(molecule: Chem.Mol, property_name: str, default=None):
    return (
        molecule.GetProp(property_name) if molecule.HasProp(property_name) else default
    )


def parse_comma_separated_property(
    molecule: Chem.Mol, property_name: str, convert_func=None
) -> List:
    property_value = get_molecule_property(molecule, property_name)
    if not property_value:
        return []

    values = property_value.split(",")
    if convert_func:
        values = [convert_func(v) if v != "None" else None for v in values]
    else:
        values = [None if v == "None" else v for v in values]

    return values


def infer_attachment_points(molecule: Chem.Mol, rgroup_indices: List[int]) -> List[int]:
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


@lru_cache
def load_monomer_library(library_path: Optional[str] = None) -> Dict:
    """Load and prepare monomer data from SDF file."""
    if library_path is None:
        library_path = str(files("helmkit.data") / "monomers.sdf")
    monomers_dict = {}
    supplier = Chem.SDMolSupplier(library_path, removeHs=False)

    for mol in supplier:
        if mol is None:
            continue

        symbol = get_molecule_property(mol, "symbol")
        if not symbol:
            continue

        rgroups = parse_comma_separated_property(mol, "m_Rgroups")
        rgroup_idx = parse_comma_separated_property(mol, "m_RgroupIdx", int)
        attachment_point_idx = infer_attachment_points(mol, rgroup_idx)

        monomers_dict[symbol] = {
            "m_romol": mol,
            "m_Rgroups": rgroups,
            "m_RgroupIdx": rgroup_idx,
            "m_attachmentPointIdx": attachment_point_idx,
            "m_type": get_molecule_property(mol, "m_type", ""),
            "m_subtype": get_molecule_property(mol, "m_subtype", ""),
            "m_abbr": get_molecule_property(mol, "m_abbr", ""),
        }

    return monomers_dict


def _create_missing_monomer(monomer_name: str) -> Dict:
    mol = Chem.MolFromSmiles(monomer_name)
    if mol is None:
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

    rgroup_idx_full = [None] * SequenceConstants.max_rgroups
    for i, (r_num, _) in enumerate(sorted_r):
        if 1 <= r_num <= SequenceConstants.max_rgroups:
            rgroup_idx_full[r_num - 1] = len(main_atoms) + i

    attachment_points = infer_attachment_points(mol, rgroup_idx_full)
    rgroup_vals = [None] * SequenceConstants.max_rgroups

    if "_R1" not in monomer_name:
        amine = Chem.MolFromSmarts("[NX3H2][#6]")
        matches = mol.GetSubstructMatches(amine)
        if len(matches) == 1:
            attachment_id, _ = matches[0]

            mol = Chem.RWMol(mol)
            mol = Chem.RWMol(mol)
            new_idx = mol.AddAtom(Chem.Atom(0))
            mol.AddBond(attachment_id, new_idx, Chem.BondType.SINGLE)
            rgroup_idx_full[0] = new_idx
            attachment_points[0] = attachment_id

    if "_R2" not in monomer_name:
        aldehide = Chem.MolFromSmarts("[CX3H1]=O")
        matches = mol.GetSubstructMatches(aldehide)
        if len(matches) == 1:
            attachment_id, _ = matches[0]

            mol = Chem.RWMol(mol)
            new_idx = mol.AddAtom(Chem.Atom(0))
            mol.AddBond(attachment_id, new_idx, Chem.BondType.SINGLE)
            rgroup_idx_full[1] = new_idx
            attachment_points[1] = attachment_id

    mol.SetProp("m_name", monomer_name)

    mol.SetProp("symbol", monomer_name)
    mol.SetProp("m_abbr", monomer_name)
    mol.SetProp("m_type", "aa")
    mol.SetProp("m_subtype", "non-natural")
    mol.SetProp("m_RgroupIdx", ",".join(map(str, rgroup_idx_full)))
    mol.SetProp("m_Rgroups", ",".join(map(str, rgroup_vals)))
    mol.SetProp("m_attachmentPointIdx", ",".join(map(str, attachment_points)))
    mol.SetProp("natAnalog", "")

    monomer = {
        "m_romol": mol,
        "m_Rgroups": rgroup_vals,
        "m_RgroupIdx": rgroup_idx_full,
        "m_attachmentPointIdx": attachment_points,
        "m_type": "aa",
        "m_subtype": "non-natural",
        "m_abbr": monomer_name,
    }
    return monomer


class Molecule:
    """Single class for HELM to RDKit Mol conversion."""

    _bracket_re = re.compile(r"{(.*?)}")
    _pipe_outside_brackets = re.compile(r"\|(?![^\[]*\])")
    _dollar_outside_brackets = re.compile(r"\$(?![^\[]*\])")

    def __init__(self, helm: str, monomer_df: Optional[Dict] = None):
        """Initialize a Molecule object from a HELM string."""
        self.mol = None
        self.offset = []
        self.bondlist = []
        self.monomers = []
        self.chain_offset = {}
        self.has_ambiguous_monomers = False

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

        polymer_sections, connection_sections = helm_parts[0], helm_parts[1]

        if not polymer_sections:
            warnings.warn(f"No simple polymers in HELM string {helm}")
            return

        self._process_polymers(polymer_sections)
        self._process_connections(connection_sections)

    def _split_helm_sections(self, helm: str) -> List:
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

        return parts

    def _split_sequence_with_brackets(self, sequence: str) -> List[str]:
        """Split a sequence into individual monomers, respecting brackets."""
        result = []
        current = ""
        bracket_depth = 0

        for char in sequence:
            if char == "[":
                bracket_depth += 1
                current += char
            elif char == "]":
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

    def _extract_chain_id(self, chain_str: str) -> Tuple[int, bool]:
        """Extract chain ID and validate chain type."""
        if chain_str.startswith("CHEM"):
            return None, False

        if not chain_str.startswith("PEPTIDE"):
            warnings.warn(f"Non-peptide chain: {chain_str}")
            return None, False

        try:
            return int(chain_str.replace("PEPTIDE", "")), True
        except ValueError:
            warnings.warn(f"Invalid chain ID: {chain_str}")
            return None, False

    def _process_monomer(
        self, monomer_name: str, chain_id: int, residue_idx: int
    ) -> Optional[Dict]:
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
            return self._process_monomer(match.group(1), chain_id, residue_idx)

        if monomer_name not in self.monomer_df:
            monomer = _create_missing_monomer(monomer_name)
            self.monomer_df[monomer_name] = monomer

        monomer_info = self.monomer_df[monomer_name]

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
            chain_id, valid = self._extract_chain_id(id_chain)
            if not valid:
                continue

            sequence = match.group(1)
            if not sequence:
                warnings.warn(f"Empty polymer: {chain}")
                continue

            residues = self._split_sequence_with_brackets(sequence)
            self.chain_offset[chain_id] = monomer_idx

            for residue_idx, monomer_name in enumerate(residues):
                monomer = self._process_monomer(monomer_name, chain_id, residue_idx)
                if not monomer:
                    continue

                self.monomers.append(monomer)

                # If this is not the first monomer, create a backbone bond to the previous one.
                if residue_idx > 0:
                    # Previous monomer is at index monomer_idx - 1, current is at monomer_idx
                    monomer1 = self.monomers[monomer_idx - 1]
                    monomer2 = monomer

                    # Standard peptide bond is between R2 of previous and R1 of current
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
                    self._mark_used_rgroup(monomer_idx - 1, attachment_point1)
                    self._mark_used_rgroup(monomer_idx, attachment_point2)

                monomer_idx += 1

    def _parse_connection(self, connection_str: str) -> Optional[Tuple]:
        """Parse a single connection string."""
        parts = connection_str.split(",")
        if len(parts) != 3:
            warnings.warn(f"Invalid connection format: {connection_str}")
            return None

        chain_id1, chain_id2, bond_spec = parts

        try:
            chain_id1 = int(chain_id1.replace("PEPTIDE", ""))
            chain_id2 = int(chain_id2.replace("PEPTIDE", ""))

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

            monomer_idx1 = self.chain_offset[chain_id1] + residue1
            monomer_idx2 = self.chain_offset[chain_id2] + residue2

            monomer1 = self.monomers[monomer_idx1]
            monomer2 = self.monomers[monomer_idx2]

            attachment_idx1 = monomer1["m_attachmentPointIdx"][rgroup1 - 1]
            if attachment_idx1 is None:
                raise ValueError(
                    f"R-group {rgroup1} is not present in monomer {monomer_idx1 + 1} ({monomer1['m_name']}). Check connections."
                )
            attachment_idx2 = monomer2["m_attachmentPointIdx"][rgroup2 - 1]
            if attachment_idx2 is None:
                raise ValueError(
                    f"R-group {rgroup2} is not present in monomer {monomer_idx2 + 1} ({monomer2['m_name']}). Check connections."
                )

            self.bondlist.append(
                [monomer_idx1, attachment_idx1, monomer_idx2, attachment_idx2]
            )

            self._mark_used_rgroup(monomer_idx1, attachment_idx1)
            self._mark_used_rgroup(monomer_idx2, attachment_idx2)

    def _mark_used_rgroup(self, monomer_idx: int, attachment_idx: int) -> None:
        """Mark an R-group as used based on its attachment point index."""
        monomer = self.monomers[monomer_idx]
        for i, idx in enumerate(monomer["m_attachmentPointIdx"]):
            if idx == attachment_idx:
                monomer["m_Rgroups"][i] = None
                break

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
        self.mol = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmarts("[#0]"))


def _init_pool(monomer_df: Dict):
    global _monomer_df
    _monomer_df = monomer_df


def _load_peptide(helm: str) -> Molecule:
    return Molecule(helm, _monomer_df)


def load_peptides_in_parallel(
    helms: List[str], monomer_df: Optional[Dict] = None, chunksize: Optional[int] = 256
) -> List[Molecule]:
    if monomer_df is None:
        monomer_df = load_monomer_library()
    with multiprocessing.Pool(initializer=_init_pool, initargs=(monomer_df,)) as pool:
        return pool.map(_load_peptide, helms, chunksize=chunksize)
