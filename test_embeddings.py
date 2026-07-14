import sys
sys.path.append(".")
from src.loader import load_model
from src.embeddings import get_embedding

if __name__ == "__main__":
    print("Testing embeddings extraction...")
    try:
        model, tokenizer = load_model("models/megamolbart")
        smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O"] # Ethanol, Aspirin
        
        # Test individual
        emb_tensor, emb_np = get_embedding("CCO", model, tokenizer)
        print(f"Single embedding shape: {emb_np.shape}")
        
        # Test batch
        batch_tensor, batch_np = get_embedding(smiles_list, model, tokenizer, normalize=True)
        print(f"Batch embedding shape: {batch_np.shape}")
        
        # Check normalization
        norms = (batch_np ** 2).sum(axis=1)
        print(f"Batch norms (should be ~1.0): {norms}")
        
        print("Embeddings test passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
