import sys
sys.path.append(".")
from src.loader import load_model

if __name__ == "__main__":
    print("Testing loader...")
    try:
        model, tokenizer = load_model("models/megamolbart")
        print("Loader test passed!")
        print("Testing tokenizer...")
        tokens = tokenizer.encode("CCO")
        print(f"Encoded 'CCO': {tokens}")
        decoded = tokenizer.decode(tokens)
        print(f"Decoded: {decoded}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
