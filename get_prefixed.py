import argparse
import sys
import random
import re
from typing import Iterable, List, Sequence, Set, Tuple, Dict, Any, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem


# ---------- Utilities ----------

def parse_index_list(text: str) -> List[int]:
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace(" ", "").replace("\t", "")
    if not s:
        return []
    return [int(x) for x in s.split(",") if x != ""]


# ---------- SMILES generation ----------

def generate_lots_of_smiles(smi: str, N: int, seed: Optional[int] = None) -> List[str]:
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
        # keep stereochemistry in the output
        new_smi = Chem.MolToSmiles(new_mol, canonical=False, isomericSmiles=True)
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
    mode: str = "all",
    max_matches: int = 1000
) -> List[Tuple[str, Set[int]]]:
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if ref_mol is None:
        raise ValueError("Invalid reference SMILES.")
    if include_h:
        ref_mol = Chem.AddHs(ref_mol)

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
            ref_to_q = {i_ref: j_q for j_q, i_ref in enumerate(match_tuple)}
            return {ref_to_q[i] for i in ref_indices_set if i in ref_to_q}

        if mode == "first":
            indices_q = mapped_set_from_match(matches[0])
        else:
            indices_q = set()
            for m in matches:
                indices_q.update(mapped_set_from_match(m))

        results.append((q_smi, indices_q))

    return results


# ---------- Assessment (find winners) ----------

def _run_len_from_zero(idxs: Iterable[int]) -> int:
    s = set(idxs)
    if 0 not in s:
        return 0
    k = 0
    while k in s:
        k += 1
    return k  # count of atoms: indices 0..k-1 present


def assess_mapped_index_sets(
    mapped_results: List[Tuple[str, Set[int]]],
    return_all_best: bool = True
) -> Dict[str, Any]:
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
        (e["smiles"], e["indices_sorted"])   # <-- was indices_sorted' )
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


# ---------- New: textual prefix that contains exactly N atoms ----------

_BOND_CHARS = set("-=#:\\/")  # characters that may precede ring digits

def _consume_ring_annots(s: str, i: int) -> int:
    """
    Starting at position i (immediately after an atom token),
    consume any number of ring-closure annotations that are attached to that atom.
    Handles optional bond char + digit, multiple digits, and %nn (two-digit ring).
    Returns the new index after consuming.
    """
    L = len(s)
    j = i
    while j < L:
        start = j

        # Optional bond char before a ring index
        if j < L and s[j] in _BOND_CHARS:
            j += 1

        # Single digit ring (0..9)
        if j < L and s[j].isdigit():
            j += 1
            continue

        # Two-digit ring: %nn
        if j + 2 < L and s[j] == '%' and s[j+1].isdigit() and s[j+2].isdigit():
            j += 3
            continue

        # No ring token consumed this iteration -> stop
        if j == start:
            break

    return j


def smiles_prefix_by_atoms(smi: str, atom_count: int) -> str:
    """
    Return the SMILES *prefix substring* that encodes exactly `atom_count` atoms,
    counting valid SMILES atom tokens (including [bracket] atoms, Cl/Br, aromatic se/as).
    The prefix *includes* any ring-closure annotations attached to the last included atom.
    It does NOT try to balance parentheses or ensure a valid standalone SMILES.
    """
    if atom_count <= 0:
        return ""

    i = 0
    atoms = 0
    L = len(smi)

    while i < L and atoms < atom_count:
        ch = smi[i]

        # 1) Bracket atom: [ ... ]
        if ch == '[':
            j = smi.find(']', i + 1)
            if j == -1:
                # Malformed; include the rest
                i = L
            else:
                i = j + 1
            atoms += 1

            # If this was the last atom we need, also include ring annotations
            if atoms == atom_count:
                i = _consume_ring_annots(smi, i)
                break
            continue

        # 2) Two-letter organic/aromatic outside brackets: Cl, Br, se, as
        two = smi[i:i+2]
        if two in ("Cl", "Br", "se", "as"):
            i += 2
            atoms += 1
            if atoms == atom_count:
                i = _consume_ring_annots(smi, i)
                break
            continue

        # 3) One-letter organic/aromatic outside brackets: B C N O P S F I b c n o p s
        if ch.isalpha():
            i += 1
            atoms += 1
            if atoms == atom_count:
                i = _consume_ring_annots(smi, i)
                break
            continue

        # 4) Non-atom characters: bonds, digits, %, branches, dot, etc. — just include
        if ch == '%' and i + 2 < L and smi[i+1].isdigit() and smi[i+2].isdigit():
            i += 3
        else:
            i += 1

    return smi[:i]


# ---------- NEW: String-safe '@' toggling guided by atom maps + robust E/Z ----------

def _toggle_at_token(body: str) -> str:
    """Flip '@' <-> '@@' once inside a bracket body (e.g., 'C@@H' -> 'C@H')."""
    if "@@" in body:
        return body.replace("@@", "@", 1)
    if "@" in body:
        return body.replace("@", "@@", 1)
    # If no '@' present, inject a single '@' after element/isotope
    m = re.match(r'^(\d+)?([A-Za-z][a-z]?)(.*)$', body)
    if m:
        return f"{m.group(1) or ''}{m.group(2)}@{m.group(3)}"
    return body + "@"


def _smiles_with_atom_maps(mol: Chem.Mol) -> str:
    # Assign temporary 1-based atom maps so RDKit brackets every atom
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    s = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return s


def _toggle_ats_in_mapped_smiles(smiles_with_maps: str, q_indices_to_flip: Set[int]) -> str:
    """
    Given a SMILES where each atom is bracketed and ends with ':map',
    flip '@' <-> '@@' only for atoms whose (map-1) is in q_indices_to_flip.
    Drop the ':map' annotations in the returned string.
    """
    def repl(m: re.Match) -> str:
        body = m.group("body")
        amap = int(m.group("map"))
        if (amap - 1) in q_indices_to_flip:
            body = _toggle_at_token(body)
        return f'[{body}]'  # remove :map
    # Pattern like [C@@H:12] or [nH:7]
    return re.sub(r'\[(?P<body>[^\[\]]*?):(?P<map>\d+)\]', repl, smiles_with_maps)


def align_stereo_like_reference(ref_smiles: str, query_smiles: str, method: str = "string") -> str:
    """
    Align the stereochemistry of query_smiles to ref_smiles.
    - method='string': tetrahedral centers flipped by toggling '@' in an atom-mapped SMILES (simple & fast);
                       double-bond E/Z fixed on the molecule.
    - method='mol'   : flip atom chirality on the molecule (more robust).
    Returns a non-canonical isomeric SMILES; if alignment fails, returns the original query_smiles.
    """
    ref = Chem.MolFromSmiles(ref_smiles)
    q = Chem.MolFromSmiles(query_smiles)
    if ref is None or q is None:
        return query_smiles

    Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
    Chem.AssignStereochemistry(q, cleanIt=True, force=True)

    # Map ref->q ignoring chirality to still get a mapping when parity differs
    mapping: Tuple[int, ...] = q.GetSubstructMatch(ref, useChirality=False)
    if not mapping or len(mapping) != ref.GetNumAtoms():
        return query_smiles

    # Determine which q atoms need flipping (CIP mismatch)
    q_atoms_to_flip: Set[int] = set()
    for r_idx, q_idx in enumerate(mapping):
        a_r = ref.GetAtomWithIdx(r_idx)
        a_q = q.GetAtomWithIdx(q_idx)
        if a_r.HasProp("_CIPCode") and a_q.HasProp("_CIPCode"):
            if a_r.GetProp("_CIPCode") != a_q.GetProp("_CIPCode"):
                q_atoms_to_flip.add(q_idx)

    # Align E/Z on the molecule (string flipping of / and \ is fragile)
    changed_ez = False
    for b_r in ref.GetBonds():
        if b_r.GetBondType() != rdchem.BondType.DOUBLE:
            continue
        s_r = b_r.GetStereo()
        if s_r not in (rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE):
            continue
        qb = (mapping[b_r.GetBeginAtomIdx()], mapping[b_r.GetEndAtomIdx()])
        b_q = q.GetBondBetweenAtoms(qb[0], qb[1])
        if b_q is None:
            continue
        if b_q.GetStereo() != s_r:
            st_r = list(b_r.GetStereoAtoms())
            if len(st_r) == 2:
                b_q.SetStereoAtoms(mapping[st_r[0]], mapping[st_r[1]])
            b_q.SetStereo(s_r)
            changed_ez = True

    if method == "mol":
        changed = changed_ez
        for idx in q_atoms_to_flip:
            a = q.GetAtomWithIdx(idx)
            tag = a.GetChiralTag()
            if tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                a.SetChiralTag(rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                changed = True
            elif tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                a.SetChiralTag(rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                changed = True
        if changed:
            Chem.AssignStereochemistry(q, cleanIt=True, force=True)
        return Chem.MolToSmiles(q, canonical=False, isomericSmiles=True)

    # method == "string": do atom flips by '@' toggles in an atom-mapped SMILES
    if q_atoms_to_flip or changed_ez:
        # Re-perceive stereo after E/Z tweak
        Chem.AssignStereochemistry(q, cleanIt=True, force=True)
        s_map = _smiles_with_atom_maps(q)
        s_toggled = _toggle_ats_in_mapped_smiles(s_map, q_atoms_to_flip)
        mq = Chem.MolFromSmiles(s_toggled)
        if mq:
            Chem.AssignStereochemistry(mq, cleanIt=True, force=True)
            return Chem.MolToSmiles(mq, canonical=False, isomericSmiles=True)

    # Fallback
    return query_smiles


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Generate randomized SMILES, map a reference index set to each, "
                    "report winners with the longest consecutive run from 0, "
                    "then align winners' stereochemistry to the reference and print prefixes."
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

    k_inclusive = summary["best"]["max_run_len"] - 1  # last atom index in the 0..k run
    k_count = summary["best"]["max_run_len"]         # number of atoms to include in prefix
    print(f"Number of prefixed SMILES: {len(winners)}")
    print("prefixed SMILES (SMILES_aligned, mapped_indices_sorted, textual_prefix_for_atoms[0..k]):")
    for smi, idxs in winners:
        # Align stereochemistry to reference before extracting the prefix
        aligned = align_stereo_like_reference(ref, smi, method="string")
        prefix = smiles_prefix_by_atoms(aligned, k_count)
        print(aligned, idxs, f"[0..{k_inclusive}] → {prefix}")


if __name__ == "__main__":
    main()
