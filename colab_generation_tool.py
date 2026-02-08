import argparse
import json
from annalog.model_handler import SMILESModelHandler
from annalog.SMILES_generator import SMILESGenerator
import torch

def main(args):
    """
    Main function to set up and generate SMILES using the API.
    """
    # Ensure prefix is either int or str
    try:
        if args.prefix.isdigit():
            args.prefix = int(args.prefix)  # Convert to int if it's a digit
    except AttributeError:
        pass  # If it's already an int, do nothing

    # Initialize the model handler and generator
    handler = SMILESModelHandler(
        src_vocab_path=args.vocab_path,
        trg_vocab_path=args.vocab_path,
        model_path=args.model_checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    generator = SMILESGenerator(handler)

    # Single SMILES input
    input_smiles = args.input_SMILES.strip()

    # Handle exploration method
    if args.exploration_method == "normal":
        # Normal generation
        generated_smiles = generator.generate_smiles(
            input_smiles=input_smiles,
            generation_number=args.generation_number,
            temperature=args.temperature,
            generation_method=args.generation_method,
            prefix=args.prefix,
            filter_invalid=args.filter_invalid
        )
        results = [(smi, prob) for smi, prob in generated_smiles]

    elif args.exploration_method == "variants":
        # Generate variants and then generate SMILES for each variant
        variants = generator.generate_variants(input_smiles, args.variant_number)
        all_generated = []
        for variant in variants:
            generated = generator.generate_smiles(
                input_smiles=variant,
                generation_number=args.generation_number,
                temperature=args.temperature,
                generation_method=args.generation_method,
                prefix=args.prefix,
                filter_invalid=args.filter_invalid
            )
            all_generated.extend(generated)
        results = [(smi, prob) for smi, prob in all_generated]

    elif args.exploration_method == "recursive":
        # Recursive generation
        current_smiles = [input_smiles]
        all_generated = []
        for _ in range(args.loops):
            next_smiles = []
            for smi in current_smiles:
                generated = generator.generate_smiles(
                    input_smiles=smi,
                    generation_number=args.generation_number,
                    temperature=args.temperature,
                    generation_method=args.generation_method,
                    prefix=args.prefix,
                    filter_invalid=args.filter_invalid
                )
                all_generated.extend(generated)
                next_smiles.extend([s[0] for s in generated])
            current_smiles = next_smiles
        results = [(smi, prob) for smi, prob in all_generated]

    else:
        raise ValueError("Invalid exploration method. Choose from 'normal', 'variants', or 'recursive'.")

    # Output results as list of tuples
    print(json.dumps(results))  # Ensure results are valid JSON

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANNalog: Generate medchem-similar analogs using a neural net trained on pairs of molecules from the same paper.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--generation_method", type=str, required=True, choices=['beam', 'BF-beam', 'sampling'],
                        help="Generation method: beam (beam search), BF-beam (best-first beam search), or sampling.")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="""Temperature to be used for the 'sampling' generation method.
                                Larger values increase the probability of choosing less likely tokens.
                                A value of 1.0 matches the prior distribution exactly while the default of 1.2
                                allows modest exploration beyond the prior without introducing unlikely tokens.
                                To explore further, consider feeding the generated SMILES into ANNalog a second (or more)
                                time using the 'recursive' exploration method rather than increasing the temperature.""")
    parser.add_argument("--prefix", required=True, help="Fixed prefix (can be int or str).")
    parser.add_argument("--filter_invalid", type=bool, default=True,
                        help="""Invalid SMILES are filtered out by default. Set to 'no' to include these in the output.
                                ANNalog uses the 'partialsmiles' library to filter invalid SMILES prefixes during
                                the generation process. This is particularly useful during beam search to avoid
                                carrying forward an invalid prefix.""")
    parser.add_argument("--generation_number", type=int, required=True, help="Number of SMILES to generate.")
    parser.add_argument("--input_SMILES", type=str, required=True, help="Source SMILES string.")
    parser.add_argument("--exploration_method", type=str, default="normal", choices=["normal", "variants", "recursive"],
                        help="""Which exploration method to use? The default is 'normal' which uses the SMILES string as presented.
                                Given that generated SMILES strings tend to preserve the prefix, 'variants' creates
                                multiple variants of the original SMILES string with different atom orders in order
                                to evenly distribute changes across the molecule (see --variant_number).
                                Specifying 'recursive' feeds the generated SMILES strings into ANNalog again
                                (and again, if --loops > 1); this helps explore more distant similarities.""")
    parser.add_argument("--variant_number", type=int, default=10, help="Number of variants to generate (used in 'variants' mode, default is 10).")
    parser.add_argument("--loops", type=int, default=1, help="Number of recursive loops (used in 'recursive' mode, default is 1).")

    args = parser.parse_args()
    main(args)
