import torch
from typing import List
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .tokenizer import MegaMolBARTTokenizer

def generate_molecules(
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    model: BartForConditionalGeneration,
    tokenizer: MegaMolBARTTokenizer,
    num_return_sequences: int = 1,
    max_length: int = 100,
    temperature: float = 1.0
) -> List[str]:
    """
    Decodes encoder hidden states back into SMILES strings.
    """
    model.eval()
    
    # Wrap the hidden states in BaseModelOutput as expected by HF generate()
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
    
    with torch.no_grad():
        output_ids = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True if temperature != 1.0 else False,
            temperature=temperature,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        
    # Decode
    smiles_list = []
    for seq in output_ids:
        # Convert tensor to list of ints
        seq_list = seq.tolist()
        smiles = tokenizer.decode(seq_list, skip_special_tokens=True)
        smiles_list.append(smiles)
        
    return smiles_list

def interpolate_molecules(
    smiles1: str,
    smiles2: str,
    num_points: int,
    model: BartForConditionalGeneration,
    tokenizer: MegaMolBARTTokenizer
) -> List[str]:
    """
    Linearly interpolates between the latent representations of two SMILES strings
    and generates the intermediate molecules.
    """
    # 1. Encode both strings to get their token IDs
    seq1 = tokenizer.encode(smiles1, add_special_tokens=True)
    seq2 = tokenizer.encode(smiles2, add_special_tokens=True)
    
    # 2. Pad to the same length so we can interpolate their sequence tensors
    max_len = max(len(seq1), len(seq2))
    
    seq1_padded = seq1 + [tokenizer.pad_token_id] * (max_len - len(seq1))
    seq2_padded = seq2 + [tokenizer.pad_token_id] * (max_len - len(seq2))
    
    mask1 = [1] * len(seq1) + [0] * (max_len - len(seq1))
    mask2 = [1] * len(seq2) + [0] * (max_len - len(seq2))
    
    # Create batch of size 2
    input_ids = torch.tensor([seq1_padded, seq2_padded], dtype=torch.long, device=model.device)
    attention_mask = torch.tensor([mask1, mask2], dtype=torch.long, device=model.device)
    
    # 3. Get encoder hidden states
    model.eval()
    with torch.no_grad():
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
    h1 = encoder_outputs.last_hidden_state[0] # Shape: (seq_len, hidden_size)
    h2 = encoder_outputs.last_hidden_state[1] # Shape: (seq_len, hidden_size)
    
    # We will use mask1 as the base mask for generation (or a combined mask)
    # Usually, padding masks should encompass the length of both, so we use a mask of all 1s up to max_len
    # for the interpolated sequences to allow the decoder to attend to the whole sequence.
    interp_mask = torch.ones((1, max_len), dtype=torch.long, device=model.device)
    
    generated_smiles = []
    
    # 4. Interpolate
    alphas = torch.linspace(0.0, 1.0, num_points)
    
    for alpha in alphas:
        # Linear interpolation: H_new = (1 - alpha) * H1 + alpha * H2
        h_interp = (1.0 - alpha.item()) * h1 + alpha.item() * h2
        # Add batch dimension
        h_interp = h_interp.unsqueeze(0)
        
        # Generate
        smiles = generate_molecules(
            encoder_hidden_states=h_interp,
            attention_mask=interp_mask,
            model=model,
            tokenizer=tokenizer,
            max_length=150
        )
        generated_smiles.extend(smiles)
        
    return generated_smiles
