# Benchmarks

## Purpose

This benchmark evaluates the **reading speed** of
[`helmkit`](https://github.com/adaliaramon/helmkit) and
[`pyPept`](https://github.com/Boehringer-Ingelheim/pyPept) for parsing HELM notation
strings into RDKit molecules. The goal is to compare their performance and usability
differences when processing peptides in HELM format.

## Feature Comparison

The libraries differ in key implementation aspects that affect speed and robustness:

| Feature                       | `pyPept`                                             | `helmkit`                |
| ----------------------------- | ---------------------------------------------------- | ------------------------ |
| **Monomer library loading**   | Reloaded for every peptide                           | Loaded once per session  |
| **Conversion path**           | HELM → BILN → Sequence → RDKit                       | HELM → RDKit             |
| **Dependencies**              | `rdkit`, `pandas`, `biopython`, `requests`, `igraph` | `rdkit`                  |
| **Monomer library input**     | Requires directory + library name                    | Requires only file path  |
| **Directory format**          | Must be a Python module (with `__init__.py`)         | No such requirement      |
| **Monomer format strictness** | Very strict (e.g., fails on parentheses, dashes)     | More tolerant            |
| **Error handling**            | Exits the process on error                           | Raises Python exceptions |

These differences influence both speed and ease of use, especially the monomer library
loading and conversion path.

## Reading Speed Results

We used the [CycPeptMPDB](http://cycpeptmpdb.com/peptides/type_PAMPA/) dataset,
comprising 7,298 cyclic peptides in HELM format. Peptides including monomers with names
that `pyPept` cannot handle were excluded (monomers with names containing whitespace,
hyphens or parenthesis), resulting in a total of 4694 peptides. We measured the total
and average parsing time:

| Tool                                | Total Time (s) | Avg Time per Peptide (s) | Peptides per Second |
|-------------------------------------|---------------:|-------------------------:|--------------------:|
| `pyPept`                            |         676.65 |                  0.14400 |                6.94 |
| `helmkit` (DB reload every peptide) |         252.67 |                  0.05383 |               18.58 |
| `helmkit`                           |           2.07 |                  0.00044 |             2267.63 |
| `helmkit` (parallel loading)        |           0.95 |                  0.00020 |             4926.11 |

`helmkit` outperforms `pyPept` by approximately 327x when loading the monomer library
once per session. When forced to reload the library for every peptide, `helmkit` is
still about 3× faster. If we use parallelized loading in `helmkit`, we achieve an
additional 2× speedup, resulting in a total speedup of approximately 712× over
`pyPept`.

## Environment

Benchmarks were run on an Intel Core i7-4790 (4 cores, 8 threads, 3.6 GHz) with 31.1 GiB
RAM and SSD storage, using Python 3.12.10 on Arch Linux (kernel 6.15.7). Key package
versions:

- `polars 1.31.0` (CSV parsing and dataframe processing)
- `rdkit 2025.3.3` (target output format and structure processing)
- `pypept 1.0.0` (commit `ade9f5840691ad1f8fa22d13939a665c25175d5a`)
- `helmkit 0.3.4` (local development version)
