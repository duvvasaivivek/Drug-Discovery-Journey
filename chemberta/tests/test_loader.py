import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_chemberta_model

def test_loader():
    print("Testing ChemBERTa loader...")
    model, tokenizer = load_chemberta_model("DeepChem/ChemBERTa-77M-MLM")
    
    assert model is not None, "Model failed to load"
    assert tokenizer is not None, "Tokenizer failed to load"
    
    tokens = tokenizer("CCO")
    assert "input_ids" in tokens, "Tokenizer failed to tokenize"
    print("Loader test passed!")

if __name__ == "__main__":
    test_loader()
