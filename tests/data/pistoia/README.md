# Pistoia Alliance HELM test cases

`positive_testcases.txt` and `negative_testcases.txt` are copied, unchanged apart from a final newline, from
the Pistoia Alliance HELM2 reference parser,
[PistoiaHELM/HELMNotationParser](https://github.com/PistoiaHELM/HELMNotationParser)
at commit `685d7ab7886e9b5cc15fb904025fe6eee703f8f6`, under the MIT licence in
`LICENSE`.

They test HELM2 *syntax*: every positive case is notation the reference parser
accepts, and every negative case is notation it rejects. Many positive cases
describe abstract structures (BLOBs, unknown monomers, ratios, polymer groups)
that are not one concrete molecule.

`tests/test_pistoia.py` also carries cases taken from the chemistry tests of
[PistoiaHELM/HELM2NotationToolkit](https://github.com/PistoiaHELM/HELM2NotationToolkit)
at commit `b2bbb17ca2b900fa4927081922aab7a45ba2db16` (`SMILESTest.java` and
`MoleculePropertyCalculatorTest.java`), also under the MIT licence.

`HELMCoreLibrary.ndjson.gz` holds the polymer type, symbol, name and SMILES of
every monomer in the HELM core monomer library,
[PistoiaHELM/HELMMonomerSets](https://github.com/PistoiaHELM/HELMMonomerSets)
`HELMCoreLibrary/HELMCoreLibrary.json` at commit
`fecc972f4707bc4e0cc34b7de3713cdba559e177`, under the MIT licence in
`LICENSE-HELMMonomerSets`. `tests/test_pistoia_core_library.py` compares the
bundled monomer library with it.
