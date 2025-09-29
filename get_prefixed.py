import argparse
import sys
import random
from typing import Iterable, List, Sequence, Set, Tuple, Dict, Any

from rdkit import Chem
from rdkit.Chem import rdmolops


# ---------- Utilities ----------

def parse_index_list(text: str) -> List[int]:
    """
    Parse an index list from CLI, accepting formats like:
      "[0, 1, 2, 3]"  or  "0,1,2,3"  or  "0 1 2 3"
    """
    s = text.strip()
    # Remove brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # Replace common separators by commas
    s = s.replace(" ", "").replace("\t", "")
    if not s:
        return []
    return [int(x) for x in s.split(",") if x != ""]


# ---------- SMILES generation ----------

def generate_lots_of_smiles(smi: str, N: int, seed: int | None = None) -> List[str]:
    """
    Generate up to N variants of a given SMILES by shuffling atom indices
    (via RenumberAtoms) and emitting non-canonical SMILES.

    Returns a list of unique SMILES (deduplicated).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    n = mol.GetNumAtoms()
    rng = random.Random(seed)

    seen = set()
    for _ in range(N):
        perm = list(range(n))
        rng.shuffle(perm)
        new_mol = rdmolops.RenumberAtoms(mol, perm)
        new_smi = Chem.MolToSmiles(new_mol, canonical=False)
        seen.add(new_smi)
    return list(seen)


# ---------- Index mapping ----------

def map_index_set_between_smiles(
    ref_smiles: str,
    ref_indices: Iterable[int],
    query_smiles_list: Sequence[str],
    *,
    include_h: bool = False,
    use_chirality: bool = True,
    mode: str = "all",          # "all" -> union over all isomorphisms; "first" -> pick first mapping
    max_matches: int = 1000
) -> List[Tuple[str, Set[int]]]:
    """
    Map a set of atom indices from a reference SMILES to each SMILES in a list.

    Returns: [(query_smiles, {indices_in_query}) ...]
      - indices_in_query contains ONLY the mapped atoms of interest.
    """
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if ref_mol is None:
        raise ValueError("Invalid reference SMILES.")
    if include_h:
        ref_mol = Chem.AddHs(ref_mol)

    # Validate ref_indices
    n_ref = ref_mol.GetNumAtoms()
    ref_indices_set = set(ref_indices)
    bad = [i for i in ref_indices_set if i < 0 or i >= n_ref]
    if bad:
        raise IndexError(f"Reference indices out of range for ref SMILES (size {n_ref}): {bad}")

    results: List[Tuple[str, Set[int]]] = []

    for q_smi in query_smiles_list:
        q_mol = Chem.MolFromSmiles(q_smi)
        if q_mol is None:
            results.append((q_smi, set()))
            continue
        if include_h:
            q_mol = Chem.AddHs(q_mol)

        # Find isomorphisms of the query within the reference
        matches = ref_mol.GetSubstructMatches(
            q_mol,
            useChirality=use_chirality,
            uniquify=True,
            maxMatches=max_matches
        )

        if not matches:
            results.append((q_smi, set()))
            continue

        def mapped_set_from_match(match_tuple) -> Set[int]:
            # match_tuple: position j is atom index in ref_mol for atom j in q_mol
            # Invert to map ref -> query
            ref_to_q = {i_ref: j_q for j_q, i_ref in enumerate(match_tuple)}
            return {ref_to_q[i] for i in ref_indices_set if i in ref_to_q}

        if mode == "first":
            indices_q = mapped_set_from_match(matches[0])
        else:
            # Union over all symmetry-distinct matches
            indices_q = set()
            for m in matches:
                indices_q.update(mapped_set_from_match(m))

        results.append((q_smi, indices_q))

    return results


# ---------- Assessment (find winners) ----------

def _run_len_from_zero(idxs: Iterable[int]) -> int:
    """
    Return the length of the longest consecutive run starting at 0.
    Example: {0,1,2,5} -> 3  (covers 0,1,2)
             {1,2,3}   -> 0  (no 0 present)
             {}        -> 0
    """
    s = set(idxs)
    if 0 not in s:
        return 0
    k = 0
    while k in s:
        k += 1
    return k  # number of consecutive integers starting at 0 (includes 0)

def assess_mapped_index_sets(
    mapped_results: List[Tuple[str, Set[int]]],
    return_all_best: bool = True
) -> Dict[str, Any]:
    """
    Summarize results; identify winners with the longest 0..k run.
    """
    per_smiles = []
    for smi, idxs in mapped_results:
        run_len = _run_len_from_zero(idxs)
        per_smiles.append({
            "smiles": smi,
            "indices_sorted": sorted(idxs),
            "has_zero": 0 in idxs,
            "run_len_from_zero": run_len
        })

    any_has_zero = any(entry["has_zero"] for entry in per_smiles)
    max_run = max((e["run_len_from_zero"] for e in per_smiles), default=0)

    winners = [
        (e["smiles"], e["indices_sorted"])
        for e in per_smiles
        if e["run_len_from_zero"] == max_run and max_run > 0
    ]
    if not return_all_best and winners:
        winners = winners[:1]

    return {
        "any_has_zero": any_has_zero,
        "per_smiles": per_smiles,
        "best": {
            "max_run_len": max_run,
            "winners": winners
        }
    }


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Generate randomized SMILES, map a reference index set to each, "
                    "and report the SMILES with the longest consecutive run starting at 0."
    )
    ap.add_argument("ref_smiles", help="Reference SMILES (whose indices you know).")
    ap.add_argument("ref_indices", help="Indices of interest, e.g. \"[0,1,2,3]\" or \"0,1,2,3\".")
    ap.add_argument("-N", type=int, default=10000, help="How many randomized SMILES to generate (default: 10000).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional).")
    ap.add_argument("--include-h", action="store_true", help="Include explicit hydrogens for matching.")
    ap.add_argument("--no-chirality", action="store_true", help="Ignore stereochemistry during matching.")
    ap.add_argument("--mode", choices=["all", "first"], default="all",
                    help="Mapping mode: union over all isomorphisms or first only.")
    ap.add_argument("--max-matches", type=int, default=1000, help="Max matches per query for isomorphism (default: 1000).")
    args = ap.parse_args()

    ref = args.ref_smiles
    wanted = set(parse_index_list(args.ref_indices))

    if not wanted:
        print("Error: ref_indices is empty after parsing.", file=sys.stderr)
        sys.exit(2)

    # 1) Generate randomized SMILES
    random_smi = generate_lots_of_smiles(ref, args.N, seed=args.seed)
    if not random_smi:
        print("No randomized SMILES generated (invalid input or N=0).", file=sys.stderr)
        sys.exit(1)

    # 2) Map the index set
    mapped = map_index_set_between_smiles(
        ref_smiles=ref,
        ref_indices=wanted,
        query_smiles_list=random_smi,
        include_h=args.include_h,
        use_chirality=not args.no_chirality,
        mode=args.mode,
        max_matches=args.max_matches
    )

    # 3) Assess and print winners
    summary = assess_mapped_index_sets(mapped, return_all_best=True)

    print(f"Any set contains 0? {summary['any_has_zero']}")
    print(f"Longest consecutive run from 0: {summary['best']['max_run_len']}")
    winners = summary["best"]["winners"]

    if not winners:
        print("No winners (no consecutive run starting at 0 found).")
        sys.exit(0)

    print(f"Number of prefixed SMILES: {len(winners)}")
    print("prefixed SMILES (SMILES, mapped_indices_sorted):")
    for smi, idxs in winners:
        print(smi, idxs)


if __name__ == "__main__":
    main()