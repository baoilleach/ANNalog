ANNalog
=======================

ANNalog, a SMILES-to-SMILES generative model for medicinal chemistry analogue design.

Introduction
-----

ANNalog is a transformer-based sequence-to-sequence (Seq2Seq) model designed to generate medicinal-chemistry-relevant analogues of an input molecule. It supports:
- local chemical-space exploration (small, SAR-like modifications), and
- scaffold hopping (changing the core scaffold while remaining chemically relevant).

The accompanying preprint describes training on pairs of molecules drawn from the same bioactivity assay (extracted from ChEMBL), Levenshtein distance–guided SMILES alignment to improve learning of transformations, and a prefix-control feature to constrain generation.

PAPER (ChemRxiv)
----------------
https://chemrxiv.org/doi/10.26434/chemrxiv-2025-9c1v6


INSTALLATION (Conda, recommended)
--------------------------------

This repository includes a conda environment file (e.g. seq2seq_environment.yml).

1) Create the environment:
conda env create -f seq2seq_environment.yml

2) Activate it (env name comes from the yml, e.g. "annalog"):
conda activate annalog

3) Install ANNalog into the environment:
pip install -e .

Note:
- If conda solving fails due to strict channel priority, try:
  conda config --set channel_priority flexible
  then re-run the environment creation.


GENERATION (generation.py)
------------------------------

generation.py generates candidate SMILES strings from an input SMILES using a trained checkpoint + vocab.

RESOURCES (checkpoint + vocab)

By default, the script looks relative to generation.py:

ckpt_and_vocab/Lev_extended.pt
ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl

If your files are elsewhere, use --resources-dir or override --checkpoint/--vocab.


### QUICK START


Single SMILES (sampling, 10 outputs):
python generation.py -i "CC(Cl)Br" -m sampling -n 10 -p 0 -f tsv -o gen_single.tsv --temperature 1.2 --seed 42

Batch file (.smi, one SMILES per line):
python generation.py -i inputs.smi -m beam -n 100 -o gen_batch.tsv


### REQUIRED ARGUMENTS


- -i, --input
  Input SMILES string OR a path to a .smi file (one SMILES per line).

- -n, --generation-number
  Number to generate (beam width or number of samples). REQUIRED.


### OPTIONAL ARGUMENTS


Generation:
- -m, --method {beam, BF-beam, sampling} (default: beam)
- --temperature FLOAT (sampling only; default: 1.2)
- --seed INT (sampling only; default: 42)
- -p, --prefix PREFIX (default: 0)
  - 0 = no prefix constraint
  - integer like 5 = use first 5 characters of the input as prefix
  - string like "CC" = literal prefix (must match the start of the input)
- -k, --keep-invalid
  Keep invalid SMILES (disables invalid filtering). By default, invalid filtering is ON.
- --max-length INT (default: 102)

Model/resources:
- --resources-dir PATH (default: <script_dir>/ckpt_and_vocab)
- --checkpoint PATH / --ckpt PATH (default: <resources-dir>/Lev_extended.pt)
- --vocab PATH (default: <resources-dir>/stereo_experiment_vocab_ttf.pkl)

Output:
- -f, --format {tsv,csv} (default: tsv)
- -o, --out PATH output path, or '-' for stdout (default: -)

Device:
- --device {cpu,cuda} force device (default: auto-detect)


### OUTPUT FORMAT


The output includes a header row with:
input_smiles, rank (1-based), generated_smiles, score
