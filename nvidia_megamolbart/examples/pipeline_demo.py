import sys
import os

# Ensure the src directory is available in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_model
from src.embeddings import get_embedding
from src.generation import interpolate_molecules
from src.utils import filter_valid_smiles, get_molecular_properties

def main():
    print("=" * 60)
    print("MegaMolBART Drug Discovery Pipeline Demo")
    print("=" * 60)

    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "megamolbart"))
    
    # ---------------------------------------------------------
    # 1. Model Loading
    # ---------------------------------------------------------
    print("\n[1/4] Loading MegaMolBART Model and Tokenizer...")
    try:
        model, tokenizer = load_model(model_dir)
    except FileNotFoundError:
        print(f"Error: Model weights not found in {model_dir}.")
        print("Please ensure you have extracted 'model_weights.ckpt' from the .nemo file.")
        return

    # ---------------------------------------------------------
    # 2. Embedding Extraction (Latent Space)
    # ---------------------------------------------------------
    print("\n[2/4] Extracting Numerical Embeddings...")
    
    # We will use two well-known drugs for this demo:
    # Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O
    # Ibuprofen: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
    smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
    smiles2 = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    
    print(f"  Molecule A (Aspirin)   : {smiles1}")
    print(f"  Molecule B (Ibuprofen) : {smiles2}")
    
    emb_tensor, emb_np = get_embedding([smiles1, smiles2], model, tokenizer, normalize=True)
    print(f"  Extracted latent vector shape: {emb_np.shape}")
    print(f"  Successfully encoded molecules into 512-dimensional space.")

    # ---------------------------------------------------------
    # 3. Novel Molecule Generation (Latent Interpolation)
    # ---------------------------------------------------------
    print("\n[3/4] Generating Novel Molecules via Latent Interpolation...")
    num_points = 10
    print(f"  Sampling {num_points} points in a straight line between Aspirin and Ibuprofen...")
    
    generated_smiles = interpolate_molecules(smiles1, smiles2, num_points=num_points, model=model, tokenizer=tokenizer)
    
    print("  Raw Output from Generative Decoder:")
    for i, s in enumerate(generated_smiles):
        alpha = i / (num_points - 1)
        print(f"    Point {i+1} (alpha={alpha:.2f}): {s}")

    # ---------------------------------------------------------
    # 4. Chemical Validation and Property Extraction
    # ---------------------------------------------------------
    print("\n[4/4] Validating Novel Molecules with RDKit...")
    valid_smiles = filter_valid_smiles(generated_smiles)
    
    print(f"  Validation Result: {len(valid_smiles)} out of {len(generated_smiles)} generated structures are chemically valid.")
    
    if len(valid_smiles) > 0:
        print("\n  Properties of Valid Discoveries:")
        for idx, valid_s in enumerate(valid_smiles):
            props = get_molecular_properties(valid_s)
            print(f"  [{idx+1}] SMILES: {valid_s}")
            print(f"      - Molecular Weight: {props['MolecularWeight']:.2f}")
            print(f"      - Lipophilicity (LogP): {props['LogP']:.2f}")
            print(f"      - H-Bond Donors: {props['NumHDonors']}")
            print(f"      - H-Bond Acceptors: {props['NumHAcceptors']}")
            print(f"      - TPSA: {props['TPSA']:.2f}\n")
    else:
        print("  No completely valid molecules generated in this short run.")
        print("  (Try increasing num_points or tuning the sampling temperature).")

    print("=" * 60)
    print("Pipeline Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
