import json
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm


def get_helm(cid: int) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
    r = requests.get(url).json()
    helm = next(
        s
        for s in r["Record"]["Section"][1]["Section"][1]["Information"]
        if s["Name"] == "HELM"
    )["Value"]["StringWithMarkup"][0]["String"]
    return helm


def main():
    data_dir = Path(__file__).parent
    input_file = data_dir / "PubChem_compound_text_peptide.csv"
    df = pl.read_csv(input_file).rename({"Compound_CID": "CID"})
    df = df[["CID", "SMILES", "InChI", "InChIKey"]]
    with open(data_dir / "pubchem.ndjson", "w") as f:
        for row in tqdm(df.iter_rows(named=True), total=df.height):
            try:
                row["HELM"] = get_helm(row["CID"])
            except (KeyError, StopIteration, IndexError):
                continue
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
