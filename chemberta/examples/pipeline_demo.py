import sys
import os

# Ensure the src directory is available in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import pipeline
from src.loader import load_chemberta_model
from src.embeddings import get_chemberta_embedding

def main():
    print("=" * 60)
    print("ChemBERTa Pipeline Demo")
    print("=" * 60)

    # 1. Load Model
    print("\n[1/3] Loading ChemBERTa Model from Hugging Face...")
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    model, tokenizer = load_chemberta_model(model_name)

    # 2. Extract Embeddings
    print("\n[2/3] Extracting Numerical Embeddings...")
    smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    smiles2 = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" # Ibuprofen
    
    print(f"  Molecule A (Aspirin)   : {smiles1}")
    print(f"  Molecule B (Ibuprofen) : {smiles2}")
    
    emb_tensor, emb_np = get_chemberta_embedding([smiles1, smiles2], model, tokenizer, normalize=True)
    print(f"  Extracted latent vector shape: {emb_np.shape}")
    print(f"  Successfully encoded molecules into {emb_np.shape[1]}-dimensional space.")

    # 3. Masked Language Modeling
    print("\n[3/3] Predicting Masked Tokens (MLM)...")
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    
    # Benzene ring with a missing carbon
    masked_smiles = "c1ccccc<mask_1>" if "<mask_1>" in tokenizer.vocab else "c1ccccc<mask>"
    if "<mask>" not in tokenizer.vocab and "<mask_1>" not in tokenizer.vocab:
        # Fallback to standard huggingface mask if tokenizer uses something else
        masked_smiles = f"c1ccccc{tokenizer.mask_token}"
    
    print(f"  Input with mask: {masked_smiles}")
    
    try:
        results = pipe(masked_smiles)
        print("  Top predictions for the masked token:")
        for res in results[:3]:
            print(f"    - Token: '{res['token_str']}' | Confidence: {res['score']:.4f}")
    except Exception as e:
        print(f"  Could not run MLM pipeline (Mask token issue): {e}")

    print("\n" + "=" * 60)
    print("Pipeline Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
