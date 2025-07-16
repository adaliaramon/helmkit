import copy
import re
import warnings
from importlib.resources import files
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from rdkit import Chem


class SequenceConstants:
    def_path = "helmkit.data"
    def_lib_filename = "monomers.sdf"
    monomer_join = "-"
    chain_separator = "."
    csv_separator = ","
    helm_polymer = "|"
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

    values = property_value.split(SequenceConstants.csv_separator)
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


def load_monomer_library(library_path: str) -> Dict:
    """Load and prepare monomer data from SDF file."""
    monomers_dict = {}
    supplier = Chem.SDMolSupplier(library_path)

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


class Molecule:
    """Single class for HELM to RDKit Mol conversion."""

    def __init__(self, helm: str, monomer_df: Optional[Dict] = None):
        """Initialize a Molecule object from a HELM string."""
        self.mol = None
        self.offset = []
        self.bondlist = []
        self.monomers = []
        self.chains = {}
        self.chain_offset = {}

        if monomer_df is None:
            default_monomer_df_filepath = files(SequenceConstants.def_path).joinpath(
                SequenceConstants.def_lib_filename
            )
            self.monomer_df = load_monomer_library(str(default_monomer_df_filepath))
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
        self._create_backbone_bonds()
        self._fix_rgroups()

    def _split_helm_sections(self, helm: str) -> List:
        """Split a HELM string into its components."""
        parts = helm.split("$", 4)
        parts.extend([""] * (5 - len(parts)))

        parts[0] = (
            parts[0].split(SequenceConstants.helm_polymer)
            if SequenceConstants.helm_polymer in parts[0]
            else [parts[0]]
        )

        if parts[1]:
            parts[1] = (
                parts[1].split(SequenceConstants.helm_polymer)
                if SequenceConstants.helm_polymer in parts[1]
                else [parts[1]]
            )
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
        monomer_name = re.sub(r"\[(.*)\]", r"\1", monomer_name)

        if monomer_name not in self.monomer_df:
            raise ValueError(f"Monomer {monomer_name} not found in monomer library")

        monomer_info = self.monomer_df[monomer_name]

        return {
            "m_name": monomer_name,
            "m_chainID": chain_id,
            "m_resID": residue_idx,
            "m_romol": monomer_info["m_romol"],
            "m_Rgroups": copy.deepcopy(monomer_info["m_Rgroups"]),
            "m_RgroupIdx": monomer_info["m_RgroupIdx"],
            "m_attachmentPointIdx": monomer_info["m_attachmentPointIdx"],
            "m_type": monomer_info["m_type"],
            "m_subtype": monomer_info["m_subtype"],
            "m_abbr": monomer_info["m_abbr"],
        }

    def _process_polymers(self, polymers: List[str]) -> None:
        """Process polymer chains from HELM."""
        monomer_idx = 0
        chain_types = []
        chain_monomer_ids = []
        pattern = re.compile(r"{(.*?)}")

        for chain in polymers:
            chain = chain.strip()

            match = pattern.search(chain)
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

            chain_monomer_ids_local = []
            monomer_types = set()

            for residue_idx, monomer_name in enumerate(residues):
                monomer = self._process_monomer(monomer_name, chain_id, residue_idx)
                if not monomer:
                    continue

                self.monomers.append(monomer)
                chain_monomer_ids_local.append(monomer_idx)
                monomer_types.add(monomer["m_type"])
                monomer_idx += 1

            if len(monomer_types) == 1:
                chain_type = "peptide" if "aa" in monomer_types else "chem"
            else:
                chain_type = "mixed"

            chain_types.append(chain_type)
            chain_monomer_ids.append(chain_monomer_ids_local)

        self.chains = {
            "s_nChains": len(polymers),
            "s_cType": chain_types,
            "s_monomerIDs": chain_monomer_ids,
        }

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
            attachment_idx2 = monomer2["m_attachmentPointIdx"][rgroup2 - 1]

            self.bondlist.append(
                [monomer_idx1, attachment_idx1, monomer_idx2, attachment_idx2]
            )

    def _create_backbone_bonds(self) -> None:
        """Create peptide backbone bonds within each chain."""
        if not self.chains:
            return

        for chain_ids in self.chains["s_monomerIDs"]:
            for i in range(len(chain_ids) - 1):
                monomer_idx1 = chain_ids[i]
                monomer_idx2 = chain_ids[i + 1]

                monomer1 = self.monomers[monomer_idx1]
                monomer2 = self.monomers[monomer_idx2]

                attachment_points1 = monomer1["m_attachmentPointIdx"]
                attachment_points2 = monomer2["m_attachmentPointIdx"]

                self.bondlist.append(
                    [
                        monomer_idx1,
                        attachment_points1[1],
                        monomer_idx2,
                        attachment_points2[0],
                    ]
                )

    def _fix_rgroups(self) -> None:
        """Mark R-groups that are used in bonds to be deleted later."""
        for bond in self.bondlist:
            monomer_idx1, attachment_idx1, monomer_idx2, attachment_idx2 = bond

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
        self._generate_atom_offsets()
        self._combine_monomers()
        self._add_bonds()
        self._process_rgroups()
        self._sanitize()

    def _generate_atom_offsets(self) -> None:
        """Generate atom offsets for each monomer in the molecule."""
        self.offset = [0]
        current_offset = 0

        for monomer in self.monomers:
            atom_count = monomer["m_romol"].GetNumAtoms()
            current_offset += atom_count
            self.offset.append(current_offset)

    def _combine_monomers(self) -> None:
        """Combine all monomers into a single molecule."""
        if not self.monomers:
            self.mol = Chem.RWMol()
            return

        combined_mol = self.monomers[0]["m_romol"]

        for i in range(1, len(self.monomers)):
            combined_mol = Chem.CombineMols(combined_mol, self.monomers[i]["m_romol"])

        self.mol = Chem.RWMol(combined_mol)

    def _add_bonds(self) -> None:
        """Add bonds between monomers based on bond list."""
        for monomer1_idx, atom1_idx, monomer2_idx, atom2_idx in self.bondlist:
            absolute_atom1_idx = self.offset[monomer1_idx] + atom1_idx
            absolute_atom2_idx = self.offset[monomer2_idx] + atom2_idx

            self.mol.AddBond(
                absolute_atom1_idx, absolute_atom2_idx, Chem.BondType.SINGLE
            )

    def _process_rgroups(self) -> None:
        """Process R-groups in the molecule, replacing or removing as needed."""
        rwmol = Chem.RWMol(self.mol)

        for idx, monomer in enumerate(self.monomers):
            rgroups = monomer["m_Rgroups"]
            rgroup_idx = monomer["m_RgroupIdx"]
            atom_offset = self.offset[idx]

            for i in range(min(len(rgroups), SequenceConstants.max_rgroups)):
                if rgroups[i] is not None:
                    self._replace_rgroup(rwmol, atom_offset, rgroup_idx[i], rgroups[i])

        self.mol = rwmol

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
        """Clean up the molecule by removing dummy atoms and sanitizing."""
        self.mol = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmarts("[#0]"))
        Chem.SanitizeMol(self.mol)
