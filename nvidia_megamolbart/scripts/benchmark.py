import sys
import os
import time
import torch

# Ensure the src directory is available in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_model
from src.embeddings import get_embedding
from src.generation import generate_molecules

def main():
    print("=========================================")
    print("MegaMolBART Benchmark Script")
    print("=========================================")
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "megamolbart"))
    
    # Benchmark 1: Model Loading
    print("[1/3] Benchmarking Model Loading...")
    t0 = time.time()
    model, tokenizer = load_model(model_dir)
    load_time = time.time() - t0
    print(f"  --> Time taken to load weights & build graph: {load_time:.2f} seconds")

    # Benchmark 2: Embedding Extraction
    print("\n[2/3] Benchmarking Embedding Extraction...")
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"] * 10 # 30 molecules
    t0 = time.time()
    emb_tensor, emb_np = get_embedding(smiles_list, model, tokenizer, normalize=True)
    emb_time = time.time() - t0
    print(f"  --> Processed {len(smiles_list)} molecules in {emb_time:.2f} seconds")
    print(f"  --> Average throughput: {len(smiles_list)/emb_time:.2f} molecules / second")

    # Benchmark 3: Molecule Generation
    print("\n[3/3] Benchmarking Decoder Generation...")
    # Generate from the first 2 embeddings in the batch
    hidden_states = emb_tensor[:2].unsqueeze(1) # Fake seq_len of 1 for raw test
    attention_mask = torch.ones((2, 1), dtype=torch.long, device=model.device)
    
    t0 = time.time()
    smiles_gen = generate_molecules(hidden_states, attention_mask, model, tokenizer, max_length=50)
    gen_time = time.time() - t0
    print(f"  --> Generated {len(smiles_gen)} molecules in {gen_time:.2f} seconds")
    print(f"  --> Average generation time: {gen_time/len(smiles_gen):.2f} seconds / molecule")
    
    print("\n=========================================")
    print("Benchmark Complete!")
    print("=========================================")

if __name__ == "__main__":
    main()
