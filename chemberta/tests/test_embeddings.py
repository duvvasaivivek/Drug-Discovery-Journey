import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_chemberta_model
from src.embeddings import get_chemberta_embedding

def test_embeddings():
    print("Testing ChemBERTa embeddings...")
    model, tokenizer = load_chemberta_model("DeepChem/ChemBERTa-77M-MLM")
    
    smiles = ["CCO", "c1ccccc1"]
    emb_tensor, emb_np = get_chemberta_embedding(smiles, model, tokenizer, normalize=True)
    
    assert emb_np.shape[0] == 2, "Batch size mismatch"
    assert emb_np.shape[1] > 0, "Embedding dimension is 0"
    print(f"Embeddings test passed! Shape: {emb_np.shape}")

if __name__ == "__main__":
    test_embeddings()
