#!/usr/bin/env python3
"""generation.py — ANNalog command-line SMILES generation (inference)

Defaults (no args needed for ckpt/vocab):
  <this_script_dir>/ckpt_and_vocab/Lev_extended.pt
  <this_script_dir>/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl

Input (choose exactly one):
  -i/--input "CC(Cl)Br"        (single SMILES)  OR
  -i/--input inputs.smi         (file; one SMILES per line)

Output:
  Default TSV to stdout
  Optional CSV via -f/--format csv

Key CLI shorthands:
  -i  input (SMILES string or .smi file path)
  -m  method (beam, BF-beam, sampling)
  -n  generation number (REQUIRED)
  -p  prefix (0, integer length, or literal string)
  -k  keep invalid (disables invalid filtering; filtering is ON by default)
  -f  output format (tsv/csv)
  -o  output path ('-' for stdout)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch

from annalog.model_handler import SMILESModelHandler
from annalog.SMILES_generator import SMILESGenerator


def _normalize_method(user_method: str) -> str:
    """Internal decoder supports: 'beam', 'BF-beam', 'sampling'."""
    m = (user_method or "").strip()
    ml = m.lower()

    aliases = {
        "beam": "beam",
        "c-beam": "beam",
        "bf-beam": "BF-beam",
        "best-first": "BF-beam",
        "sampling": "sampling",
        "sample": "sampling",
    }
    if m == "BF-beam":
        return "BF-beam"
    if ml in aliases:
        return aliases[ml]
    raise ValueError(
        "Invalid --method. Use: beam, BF-beam, sampling (aliases: C-beam, sample)."
    )


def _parse_prefix(prefix: str) -> Union[int, str]:
    """Prefix is optional.

    - digits -> int
    - otherwise -> string prefix (must match input SMILES start)
    """
    if prefix is None:
        return 0
    p = str(prefix).strip()
    if p == "" or p == "0":
        return 0
    if p.isdigit():
        return int(p)
    return p


def _read_inputs_cli(
    input_any: Optional[str],
    input_smiles: Optional[str],
    input_file: Optional[str],
) -> List[str]:
    """Resolve CLI input.

    Priority:
      1) -i/--input: if it points to an existing file -> read SMILES lines.
         otherwise -> treat as a single SMILES string.
      2) legacy flags: --input-smiles / --input-file.

    Notes:
      - Blank lines in files are ignored.
      - If -i ends with a typical SMILES-file extension (.smi/.smiles/.txt)
        but the file does NOT exist, we raise FileNotFoundError.
        (We do NOT use '/' or '\\' heuristics because SMILES can contain them.)
    """
    # Prefer unified -i/--input if present
    if input_any is not None:
        s = input_any.strip()
        if not s:
            raise ValueError("-i/--input is empty")

        p = Path(s)
        if p.exists() and p.is_file():
            inputs: List[str] = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    smi = line.strip()
                    if smi:
                        inputs.append(smi)
            if not inputs:
                raise ValueError(f"No SMILES found in file: {p}")
            return inputs

        # If it looks like a file path by extension but doesn't exist, fail loudly.
        if p.suffix.lower() in {".smi", ".smiles", ".txt"}:
            raise FileNotFoundError(f"Input file not found: {s}")

        # Otherwise treat as SMILES.
        return [s]

    # Legacy flags
    if input_smiles:
        s = input_smiles.strip()
        if not s:
            raise ValueError("--input-smiles is empty")
        return [s]

    if not input_file:
        raise ValueError("Provide either -i/--input, --input-smiles, or --input-file")

    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    inputs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                inputs.append(s)

    if not inputs:
        raise ValueError(f"No SMILES found in file: {input_file}")
    return inputs


def main(argv: Optional[List[str]] = None) -> int:
    # ---- Defaults based on script location ----
    script_dir = Path(__file__).resolve().parent
    default_resources_dir = script_dir / "ckpt_and_vocab"
    default_checkpoint = default_resources_dir / "Lev_extended.pt"
    default_vocab = default_resources_dir / "stereo_experiment_vocab_ttf.pkl"

    parser = argparse.ArgumentParser(
        description="Generate SMILES with ANNalog (CLI). Outputs TSV by default.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Inputs (choose one)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "-i",
        "--input",
        dest="input_any",
        type=str,
        help="Input SMILES OR path to a .smi file (one SMILES per line).",
    )
    # Legacy flags (optional; keep for backwards compatibility)
    inp.add_argument("--input-smiles", dest="input_smiles", type=str, help="(Legacy) Single SMILES.")
    inp.add_argument("--input-file", dest="input_file", type=str, help="(Legacy) .smi file path.")

    # Resources
    parser.add_argument(
        "--resources-dir",
        type=str,
        default=str(default_resources_dir),
        help="Folder containing checkpoint + vocab.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        default=str(default_checkpoint),
        help="Model checkpoint (.pt).",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=str(default_vocab),
        help="Vocab pickle (torchtext-free exported dict PKL).",
    )

    # Generation options
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="beam",
        help="Decoding method: beam, BF-beam, sampling. Default: beam",
    )
    parser.add_argument(
        "-n",
        "--generation-number",
        "--n",
        dest="generation_number",
        type=int,
        required=True,
        help="Number to generate (beam width or sample count). REQUIRED.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Sampling temperature (used only when --method sampling). Default: 1.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used only when --method sampling). Default: 42",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="0",
        help="Optional prefix: integer (chars) or string. Default: 0 (no prefix)",
    )
    # Default invalid filtering ON. -k flips to keep invalid (disable filtering).
    parser.add_argument(
        "-k",
        "--keep-invalid",
        action="store_true",
        help="Keep invalid SMILES (DISABLE filtering). Default: filtering is ON.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=102,
        help="Max sequence length. Default: 102",
    )

    # Output
    parser.add_argument(
        "-f",
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format. Default: tsv",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="-",
        help="Output path. Default: '-' (stdout)",
    )

    # Device
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force device. Default: auto",
    )

    args = parser.parse_args(argv)

    # If user changed resources-dir, and did NOT override checkpoint/vocab,
    # automatically point them to the filenames inside the new resources dir.
    res_dir = Path(args.resources_dir).expanduser().resolve()
    if Path(args.checkpoint).expanduser().resolve() == default_checkpoint.resolve():
        args.checkpoint = str(res_dir / "Lev_extended.pt")
    if Path(args.vocab).expanduser().resolve() == default_vocab.resolve():
        args.vocab = str(res_dir / "stereo_experiment_vocab_ttf.pkl")

    # Validate existence
    ckpt_path = Path(args.checkpoint).expanduser()
    vocab_path = Path(args.vocab).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    method = _normalize_method(args.method)
    prefix = _parse_prefix(args.prefix)
    inputs = _read_inputs_cli(args.input_any, args.input_smiles, args.input_file)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    handler = SMILESModelHandler(
        src_vocab_path=str(vocab_path),
        trg_vocab_path=str(vocab_path),
        model_path=str(ckpt_path),
        device=device,
        max_length=args.max_length,
    )
    generator = SMILESGenerator(handler)

    out_fh = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8", newline="")
    try:
        delimiter = "\t" if args.format == "tsv" else ","
        writer = csv.writer(out_fh, delimiter=delimiter)

        writer.writerow(["input_smiles", "rank", "generated_smiles", "score"])

        for in_smi in inputs:
            generated = generator.generate_smiles(
                input_smiles=in_smi,
                generation_number=args.generation_number,
                generation_method=method,
                temperature=args.temperature,
                prefix=prefix,
                # Filtering is ON by default; -k disables it.
                filter_invalid=(not args.keep_invalid),
                seed=args.seed,
            )
            for rank, (gen_smi, score) in enumerate(generated, start=1):
                writer.writerow([in_smi, rank, gen_smi, score])

    finally:
        if out_fh is not sys.stdout:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())