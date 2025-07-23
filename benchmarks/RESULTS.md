## Benchmark Results

| Tool                                | Total Time (s) | Avg Time per Peptide (s) |
|-------------------------------------|----------------|--------------------------|
| `pyPept`                            | 676.65         | 0.144152                 |
| `helmkit`                           | 3.60           | 0.000767                 |
| `helmkit` (DB reload every peptide) | 261.81         | 0.055776                 |

## Environment

Benchmarks were performed on a system running Arch Linux (kernel 6.15.7) with an Intel
Core i7-4790 CPU (4 cores, 8 threads, 3.6 GHz base clock) and 31 GiB of RAM. All data
was stored on SSDs, and no GPU acceleration was used.

The benchmarks were executed using Python 3.12.10, built with Clang 20.1.0, and ran in a
standard Linux environment.

Key package versions used in the benchmarks include:

- `polars 1.31.0` – used for peptide input parsing
- `rdkit 2025.3.3` – used for cheminformatics and structure processing
- `pypept 1.0.0` – installed from Git at commit `ade9f5840691ad1f8fa22d13939a665c25175d5a`
- `helmkit 0.1.0` – local development version
