import sys
sys.path.append(".")
from src.loader import load_model
from src.generation import interpolate_molecules

if __name__ == "__main__":
    print("Testing molecule generation and interpolation...")
    try:
        model, tokenizer = load_model("models/megamolbart")
        
        smiles1 = "CCO" # Ethanol
        smiles2 = "c1ccccc1" # Benzene
        
        print(f"Interpolating between {smiles1} and {smiles2} with 5 points...")
        generated = interpolate_molecules(smiles1, smiles2, 5, model, tokenizer)
        
        for i, s in enumerate(generated):
            print(f"Point {i} (alpha={i/4}): {s}")
            
        print("Generation test passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
