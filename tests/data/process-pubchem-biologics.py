import polars as pl


def main():
    # grep -P '^\d+\t5\tPEPTIDE' CID-Biologics.tsv > CID-Peptides.tsv
    df = pl.read_csv(
        "CID-Peptides.tsv",
        separator="\t",
        has_header=False,
        columns=[0, 2],
        new_columns=["CID", "HELM"],
    )
    df_2 = pl.read_csv(
        "PubChem_compound_cache_yI9ul0bDI38UVSFMozRoasCThvPhmP8MhSnkQJ449kGeIco.csv",
        columns=["Compound_CID", "SMILES", "InChI"],
        new_columns=["CID"],
    )
    df = df.join(df_2, on="CID")
    df.write_csv("CID-Peptides.csv")


if __name__ == "__main__":
    main()
