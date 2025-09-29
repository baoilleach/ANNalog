import argparse
import hashlib
import os
import sys
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

def draw_with_indices_png(smiles: str, outfile: str, size=(600, 450), include_h: bool = False, bond_indices: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    if include_h:
        mol = Chem.AddHs(mol)

    rdDepictor.Compute2DCoords(mol)

    w, h = size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)  # PNG only
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    if bond_indices:
        opts.addBondIndices = True

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    data = drawer.GetDrawingText()  # bytes for Cairo/PNG
    with open(outfile, "wb") as f:
        f.write(data)
    return outfile

def main():
    p = argparse.ArgumentParser(description="Draw a 2D PNG with atom indices from a SMILES.")
    p.add_argument("smiles", help="Input SMILES string.")
    p.add_argument("outdir", help="Directory to write the PNG.")
    p.add_argument("-o", "--outfile", help="Output filename (must be .png). If omitted, a name is generated.")
    p.add_argument("--width", type=int, default=600, help="Image width (px).")
    p.add_argument("--height", type=int, default=450, help="Image height (px).")
    p.add_argument("--include-h", action="store_true", help="Draw explicit hydrogens.")
    p.add_argument("--bond-indices", action="store_true", help="Also label bond indices.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.outfile:
        if not args.outfile.lower().endswith(".png"):
            print("Error: --outfile must end with .png", file=sys.stderr)
            sys.exit(2)
        filename = args.outfile
    else:
        digest = hashlib.md5(args.smiles.encode("utf-8")).hexdigest()[:8]
        filename = f"mol_{digest}.png"

    outpath = os.path.join(args.outdir, filename)

    try:
        saved = draw_with_indices_png(
            args.smiles,
            outpath,
            size=(args.width, args.height),
            include_h=args.include_h,
            bond_indices=args.bond_indices,
        )
        print(f"Saved: {saved}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()