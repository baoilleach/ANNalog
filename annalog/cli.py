#!/usr/bin/env python3
"""annalog/cli.py — ANNalog command-line SMILES generation (inference)

This is the package-installed CLI version of generation.py.

Defaults (no args needed for ckpt/vocab):
  Uses packaged resources:
    annalog/ckpt_and_vocab/Lev_extended.pt
    annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl

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
  -e  exploration method (normal, variants, recursive)  [default: normal]
  -p  prefix (0, integer length, or literal string)
  -k  keep invalid (disables invalid filtering; filtering is ON by default)
  -f  output format (tsv/csv)
  -o  output path ('-' for stdout)

Exploration modes:
  normal:
    Generate directly from the input SMILES.

  variants:
    Generate --variant-number variants of the input SMILES, then generate from each variant.

  recursive:
    Run --loops rounds. Each round generates from the previous round’s outputs.
"""

import argparse
import csv
import sys
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from typing import List, Optional, Union

import torch

from .model_handler import SMILESModelHandler
from .SMILES_generator import SMILESGenerator


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
    # ---- Packaged defaults (inside installed annalog) ----
    default_resources_dir = files("annalog") / "ckpt_and_vocab"
    default_checkpoint_res = default_resources_dir / "Lev_extended.pt"
    default_vocab_res = default_resources_dir / "stereo_experiment_vocab_ttf.pkl"

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

    # Resources (allow override)
    parser.add_argument(
        "--resources-dir",
        type=str,
        default=None,
        help="Folder containing checkpoint + vocab. If omitted, uses packaged ckpt_and_vocab.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        default=None,
        help="Model checkpoint (.pt). If omitted, uses packaged Lev_extended.pt.",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Vocab pickle. If omitted, uses packaged stereo_experiment_vocab_ttf.pkl.",
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

    # Exploration options
    parser.add_argument(
        "-e",
        "--exploration-method",
        dest="exploration_method",
        choices=["normal", "variants", "recursive"],
        default="normal",
        help="Exploration method: normal, variants, recursive. Default: normal",
    )
    parser.add_argument(
        "--variant-number",
        type=int,
        default=10,
        help="Number of variants to generate (used only when --exploration-method variants). Default: 10",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of recursive loops (used only when --exploration-method recursive). Default: 1",
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

    # Validate exploration-specific args
    if args.variant_number < 1:
        raise ValueError("--variant-number must be >= 1")
    if args.loops < 1:
        raise ValueError("--loops must be >= 1")

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    method = _normalize_method(args.method)
    prefix = _parse_prefix(args.prefix)
    inputs = _read_inputs_cli(args.input_any, args.input_smiles, args.input_file)

    # ---- Resolve ckpt/vocab paths (packaged by default) ----
    # Use ExitStack so packaged files extracted via as_file stay valid during runtime.
    with ExitStack() as stack:
        # If --resources-dir is provided, it becomes the base for default filenames
        if args.resources_dir is not None:
            res_dir = Path(args.resources_dir).expanduser().resolve()
            default_ckpt_path = res_dir / "Lev_extended.pt"
            default_vocab_path = res_dir / "stereo_experiment_vocab_ttf.pkl"
            ckpt_path = Path(args.checkpoint).expanduser() if args.checkpoint else default_ckpt_path
            vocab_path = Path(args.vocab).expanduser() if args.vocab else default_vocab_path
        else:
            # Packaged defaults
            ckpt_path = stack.enter_context(as_file(default_checkpoint_res))
            vocab_path = stack.enter_context(as_file(default_vocab_res))

            # If user explicitly provided checkpoint/vocab, prefer those
            if args.checkpoint:
                ckpt_path = Path(args.checkpoint).expanduser()
            if args.vocab:
                vocab_path = Path(args.vocab).expanduser()

        # Validate existence
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocab not found: {vocab_path}")

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
                exploration = args.exploration_method
                out_rank = 1

                if exploration == "normal":
                    generated = generator.generate_smiles(
                        input_smiles=in_smi,
                        generation_number=args.generation_number,
                        generation_method=method,
                        temperature=args.temperature,
                        prefix=prefix,
                        filter_invalid=(not args.keep_invalid),
                        seed=args.seed,
                    )
                    for gen_smi, score in generated:
                        writer.writerow([in_smi, out_rank, gen_smi, score])
                        out_rank += 1

                elif exploration == "variants":
                    try:
                        variants = generator.generate_variants(in_smi, args.variant_number)
                    except Exception as e:
                        print(
                            f"[WARN] generate_variants failed for input {in_smi!r}: {e}",
                            file=sys.stderr,
                        )
                        variants = []

                    for variant in variants:
                        try:
                            generated = generator.generate_smiles(
                                input_smiles=variant,
                                generation_number=args.generation_number,
                                generation_method=method,
                                temperature=args.temperature,
                                prefix=prefix,
                                filter_invalid=(not args.keep_invalid),
                                seed=args.seed,
                            )
                        except Exception as e:
                            print(
                                f"[WARN] generate_smiles failed for variant {variant!r} (root {in_smi!r}): {e}",
                                file=sys.stderr,
                            )
                            continue

                        for gen_smi, score in generated:
                            writer.writerow([in_smi, out_rank, gen_smi, score])
                            out_rank += 1

                elif exploration == "recursive":
                    current_smiles: List[str] = [in_smi]

                    for loop_idx in range(args.loops):
                        next_smiles: List[str] = []

                        for parent in current_smiles:
                            try:
                                generated = generator.generate_smiles(
                                    input_smiles=parent,
                                    generation_number=args.generation_number,
                                    generation_method=method,
                                    temperature=args.temperature,
                                    prefix=prefix,
                                    filter_invalid=(not args.keep_invalid),
                                    seed=args.seed,
                                )
                            except Exception as e:
                                print(
                                    f"[WARN] generate_smiles failed in loop {loop_idx + 1} for parent {parent!r} (root {in_smi!r}): {e}",
                                    file=sys.stderr,
                                )
                                continue

                            for gen_smi, score in generated:
                                writer.writerow([in_smi, out_rank, gen_smi, score])
                                out_rank += 1
                                next_smiles.append(gen_smi)

                        current_smiles = next_smiles

                else:
                    raise ValueError(
                        "Invalid exploration method. Choose from: normal, variants, recursive."
                    )

        finally:
            if out_fh is not sys.stdout:
                out_fh.close()

    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
