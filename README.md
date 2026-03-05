ANNalog (torchtext-free)
=======================

Torchtext-free version of the ANNalog SMILES generative model (CLI inference).


INSTALL
-------

1) Clone the torchtext_free branch:
git clone --depth 1 --branch torchtext_free --single-branch https://github.com/DVNecromancer/ANNalog.git

2) Install (run inside the repo folder):
cd ANNalog
pip install -e .


GENERATION (generation.py)
--------------------------

Quick start (single SMILES):
python generation.py --input-smiles "CC(Cl)Br" --method beam --n 100 --out gen.tsv

Quick start (batch file; one SMILES per line):
python generation.py --input-file inputs.smi --method beam --n 100 --out gen.tsv


REQUIRED INPUT (choose exactly ONE)
-----------------------------------

1) --input-smiles "SMILES"
   - A single SMILES string.

OR

2) --input-file PATH
   - Path to a .smi text file containing one SMILES per line.


OPTIONAL INPUTS
---------------

Model/resources:
- --resources-dir PATH
  Folder containing checkpoint + vocab.
  Default: <script_dir>/ckpt_and_vocab

- --checkpoint PATH  (alias: --ckpt PATH)
  Model checkpoint (.pt).
  Default: <resources-dir>/Lev_extended.pt

- --vocab PATH
  Vocab pickle (.pkl).
  Default: <resources-dir>/stereo_experiment_vocab_ttf.pkl


Generation settings:
- --method beam | BF-beam | sampling
  Default: beam

- --n INT
  Number to generate.
  Default: 100

- --temperature FLOAT
  Sampling temperature (used only when --method sampling).
  Default: 1.2

- --prefix PREFIX
  Prefix constraint: 0 (none), integer, or string prefix.
  Default: 0

- --filter-invalid
  If set, filters invalid SMILES during decoding.
  Default: off

- --max-length INT
  Max sequence length.
  Default: 102


Output:
- --format tsv | csv
  Default: tsv

- --out PATH
  Output file path, or '-' for stdout.
  Default: -


Device:
- --device cpu | cuda
  Force device (otherwise auto-detect).
  Default: auto-detect
