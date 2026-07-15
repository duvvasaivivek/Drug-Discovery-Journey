import sys
import torch
sys.path.append(".")
from src.loader import load_model

model, tokenizer = load_model("models/megamolbart")
smiles = "CCO"
tokens = tokenizer.encode(smiles)
input_ids = torch.tensor([tokens], dtype=torch.long)

print(f"Input SMILES: {smiles}")
print(f"Input IDs: {input_ids.tolist()}")

model.eval()
with torch.no_grad():
    out = model.generate(
        input_ids=input_ids,
        max_length=20,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

print(f"Output IDs: {out.tolist()}")
print(f"Reconstructed: {tokenizer.decode(out[0].tolist(), skip_special_tokens=True)}")
