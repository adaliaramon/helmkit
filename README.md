# helmkit

A Python library for converting HELM (Hierarchical Editing Language for Macromolecules) notation to RDKit molecules.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Installation](#installation)
- [Quick Example](#quick-example)
- [Understanding HELM Notation](#understanding-helm-notation)
- [Mapping Atoms and Bonds to Monomers](#mapping-atoms-and-bonds-to-monomers)
- [Using Custom Monomer Data](#using-custom-monomer-data)
- [SDF File Structure Requirements](#sdf-file-structure-requirements)
- [Parallel Processing of Peptides](#parallel-processing-of-peptides)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)

## Basic Usage

```python
from helmkit import Molecule

# Create a molecule from a HELM string
helm_string = "PEPTIDE1{A.R.G}$$$$"
molecule = Molecule(helm_string)

# Access the RDKit molecule object
rdkit_mol = molecule.mol
```

## Installation

To install `helmkit`, you can use either [`uv`](https://github.com/astral-sh/uv) or `pip`.

### With [`uv`](https://github.com/astral-sh/uv)

```bash
uv pip install helmkit
```

or if you have added it as a dependency to your `pyproject.toml`:

```bash
uv add helmkit
```

### Without `uv`

```bash
pip install helmkit
```

## Quick Example

```python
from helmkit import Molecule
from rdkit.Chem import AllChem, Draw

# Create a simple tripeptide (Ala-Arg-Gly)
molecule = Molecule("PEPTIDE1{A.R.G}$$$$")

# Generate 2D coordinates for visualization
AllChem.Compute2DCoords(molecule.mol)

# Save the image
img = Draw.MolToImage(molecule.mol)
img.save("tripeptide.png")
```

## Understanding HELM Notation

HELM (Hierarchical Editing Language for Macromolecules) is a notation for representing complex biomolecules. A basic HELM string has the following format:

```
PEPTIDE1{A.R.G}|PEPTIDE2{S.G.C}$PEPTIDE1,PEPTIDE2,1:R1-3:R3$$$V2.0
```

Where:
- `PEPTIDE1{A.R.G}` defines the first chain (a peptide with amino acids A, R, G)
- `PEPTIDE2{S.G.C}` defines the second chain
- `PEPTIDE1,PEPTIDE2,1:R1-3:R3` defines a connection between the chains (R1 of residue 1 in PEPTIDE1 connects to R3 of residue 3 in PEPTIDE2)
- `$` characters separate different sections of the HELM string

## Mapping Atoms and Bonds to Monomers

Besides the RDKit molecule, a `Molecule` records which monomer every atom came
from and which bonds join one monomer to another. Together they describe the
molecule as a graph of monomers:

- `molecule.monomers` lists the monomers in the order they appear in the HELM
  string, chain by chain. In an RNA chain the sugar, base and phosphate of a
  nucleotide are separate monomers, so `RNA1{R(A)P.R(C)}` has the five monomers
  `R`, `A`, `P`, `R` and `C`. `monomer["m_abbr"]` is the name of a monomer: its
  `m_abbr` in the monomer library, or the SMILES string of an inline monomer.
- `molecule.monomer_indices` gives, for every atom of `molecule.mol`, the index
  in `molecule.monomers` of the monomer it came from. Each monomer's atoms are
  contiguous and in the same order as the monomers.
- `molecule.bond_indices` gives the index in `molecule.mol` of every bond
  between two monomers: the backbone bonds along each chain, followed by the
  bonds in the connection section.

```python
from helmkit import Molecule

# A cyclic peptide: the side chains of the two cysteines form a disulfide bond
molecule = Molecule("PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$")
mol = molecule.mol

names = [monomer["m_abbr"] for monomer in molecule.monomers]
monomer_of = molecule.monomer_indices

edges = []
for bond_idx in molecule.bond_indices:
    bond = mol.GetBondWithIdx(bond_idx)
    edges.append((monomer_of[bond.GetBeginAtomIdx()], monomer_of[bond.GetEndAtomIdx()]))

print(names)  # ['A', 'C', 'G', 'C']
print(edges)  # [(0, 1), (1, 2), (2, 3), (1, 3)]
```

Removing the bonds in `bond_indices`, for instance with
`Chem.FragmentOnBonds(mol, molecule.bond_indices)`, splits the molecule into
one fragment per monomer, unless a connection bonds a monomer to itself or a
monomer is a salt: a counter-ion, such as the sodium of
`[Na+].[O-]P([*])([*])=O`, is a fragment of its own but still belongs to its
monomer in `monomer_indices`.

## Using Custom Monomer Data

By default, helmkit uses the monomer data in `helmkit/data/monomers.sdf`. To use a custom SDF file:

```python
from helmkit import Molecule, load_monomer_library

# Load your custom monomer data
custom_sdf_path = "/path/to/your/custom_monomers.sdf"
custom_monomers = load_monomer_library(custom_sdf_path)

# Create molecule with custom monomer data
molecule = Molecule("PEPTIDE1{A.R.G}$$$$", monomer_df=custom_monomers)
```

## SDF File Structure Requirements

The SDF file containing monomer data must have the following properties for each molecule:

### Required Properties:
- `symbol`: A unique identifier for the monomer (e.g., "A" for alanine)
- `m_RgroupIdx`: Comma-separated list of R-group atom indices (e.g., "1,2,None,None")

### Optional Properties:
- `m_Rgroups`: Comma-separated list of R-group types (e.g., "H,OH,None,None")
- `m_type`: Monomer type (e.g., "aa" for amino acid)
- `m_subtype`: Monomer subtype
- `m_abbr`: Monomer abbreviation

### Example SDF Entry:

```
Your molecule atom data here...
...

> <symbol>
A

> <m_Rgroups>
H,OH,None,None

> <m_RgroupIdx>
1,2,None,None

> <m_type>
aa

> <m_subtype>
natural

> <m_abbr>
Ala

$$$$
```

## Parallel Processing of Peptides

For workflows involving a large number of peptides, `helmkit` provides a function to process them in parallel, significantly improving performance.

```python
from helmkit import load_monomer_library
from helmkit import load_in_parallel

# Load your custom monomer data (optional)
custom_sdf_path = "/path/to/your/custom_monomers.sdf"
monomer_db = load_monomer_library(custom_sdf_path)

# A list of HELM strings
helm_strings = ["PEPTIDE1{A.R.G}$$$$", "PEPTIDE1{S.G.T}$$$$"]

# Process peptides in parallel
molecules = load_in_parallel(helm_strings, monomer_db)
```

## Development Setup

To set up a development environment, first clone the repository.
Then, from the root of the repository, use `uv` to sync the environment:

```bash
uv sync -U
```

## Running Tests

To run the test suite, execute `pytest` from the root of the repository:

```bash
pytest
```
