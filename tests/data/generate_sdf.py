import polars as pl
from rdkit import Chem

from pyPept import SequenceConstants

# Some monomers were manually edited because they had more R-groups than expected
# meK, Me_dK
# Changes were made to the original CSV (e.g., changed H in R3 by -)


def main():
    df = pl.read_csv("monomers.csv", null_values=["N.D"])
    with Chem.SDWriter("monomers.sdf") as writer:
        for row in df.iter_rows(named=True):
            mol = Chem.MolFromSmiles(row["CXSMILES"])

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

            rgroup_vals = [
                None if row.get(f"R{i + 1}") == "-" else row.get(f"R{i + 1}")
                for i in range(SequenceConstants.max_rgroups)
            ]

            if row["Compound_Name"]:
                mol.SetProp("m_name", row["Compound_Name"])

            mol.SetProp("symbol", row["Symbol"])
            mol.SetProp("m_abbr", row["Symbol"])
            mol.SetProp("m_type", "aa")
            mol.SetProp("m_subtype", "non-natural")
            mol.SetProp("m_RgroupIdx", ",".join(map(str, rgroup_idx_full)))
            mol.SetProp("m_Rgroups", ",".join(map(str, rgroup_vals)))
            mol.SetProp("natAnalog", row["Natural_Analog"])

            writer.write(mol)


if __name__ == "__main__":
    main()
