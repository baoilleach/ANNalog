#!/usr/bin/env python3
"""
generation.py — ANNalog command-line SMILES generation (inference)

Defaults (no args needed for ckpt/vocab):
  <this_script_dir>/ckpt_and_vocab/Lev_extended.pt
  <this_script_dir>/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl

Input:
  --input-smiles "CC(Cl)Br"        (single SMILES)
  --input-file inputs.smi          (one SMILES per line)

Output:
  Default TSV to stdout
  Optional CSV via --format csv
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
    raise ValueError("Invalid --method. Use: beam, BF-beam, sampling (aliases: C-beam, sample).")


def _parse_prefix(prefix: str) -> Union[int, str]:
    """
    Prefix is optional.
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


def _read_inputs(input_smiles: Optional[str], input_file: Optional[str]) -> List[str]:
    if input_smiles:
        s = input_smiles.strip()
        if not s:
            raise ValueError("--input-smiles is empty")
        return [s]

    if not input_file:
        raise ValueError("Provide either --input-smiles or --input-file")

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
    inp.add_argument("--input-smiles", type=str, help="Single input SMILES string.")
    inp.add_argument("--input-file", type=str, help="Path to a .smi file (one SMILES per line).")

    # Resources (defaults are REAL paths, not None)
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
        help="Vocab pickle (torchtext-free exported dict PKL, e.g. *.nott.pkl).",
    )

    # Generation options
    parser.add_argument("--method", type=str, default="beam",
                        help="Decoding method: beam, BF-beam, sampling. Default: beam")
    parser.add_argument("--n", type=int, default=100,
                        help="Number to generate (beam width or sample count). Default: 100")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Sampling temperature (used only when --method sampling). Default: 1.2")
    parser.add_argument("--prefix", type=str, default="0",
                        help="Optional prefix: integer (chars) or string. Default: 0 (no prefix)")
    parser.add_argument("--filter-invalid", action="store_true",
                        help="Filter invalid SMILES during decoding (slower). Default: off")
    parser.add_argument("--max-length", type=int, default=102,
                        help="Max sequence length. Default: 102")

    # Output
    parser.add_argument("--format", choices=["tsv", "csv"], default="tsv",
                        help="Output format. Default: tsv")
    parser.add_argument("--out", type=str, default="-",
                        help="Output path. Default: '-' (stdout)")

    # Device
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Force device. Default: auto")

    args = parser.parse_args(argv)

    # If user changed resources-dir, and did NOT override checkpoint/vocab,
    # automatically point them to the filenames inside the new resources dir.
    # (We detect "not overridden" by seeing if the arg equals the old default.)
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
    inputs = _read_inputs(args.input_smiles, args.input_file)

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
                generation_number=args.n,
                generation_method=method,
                temperature=args.temperature,
                prefix=prefix,
                filter_invalid=args.filter_invalid,
            )
            for rank, (gen_smi, score) in enumerate(generated, start=1):
                writer.writerow([in_smi, rank, gen_smi, score])

    finally:
        if out_fh is not sys.stdout:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())