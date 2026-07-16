import torch
import numpy as np
from typing import Union, List, Tuple
from transformers import BartForConditionalGeneration
from .tokenizer import MegaMolBARTTokenizer

def get_embedding(
    smiles: Union[str, List[str]], 
    model: BartForConditionalGeneration, 
    tokenizer: MegaMolBARTTokenizer,
    normalize: bool = False
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extracts numerical embeddings for one or more SMILES strings using MegaMolBART's encoder.
    
    Args:
        smiles (str or List[str]): The input SMILES string(s).
        model (BartForConditionalGeneration): The loaded MegaMolBART model.
        tokenizer (MegaMolBARTTokenizer): The loaded MegaMolBART tokenizer.
        normalize (bool): Whether to L2 normalize the resulting embeddings.
        
    Returns:
        torch.Tensor: The embeddings as a PyTorch tensor.
        np.ndarray: The embeddings as a NumPy array.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
        
    # Tokenize and encode
    token_ids_list = [tokenizer.encode(s, add_special_tokens=True) for s in smiles]
    
    # Pad sequences for batch inference
    max_len = max(len(seq) for seq in token_ids_list)
    padded_input_ids = []
    attention_masks = []
    
    for seq in token_ids_list:
        pad_len = max_len - len(seq)
        padded_seq = seq + [tokenizer.pad_token_id] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
        
        padded_input_ids.append(padded_seq)
        attention_masks.append(mask)
        
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=model.device)
    
    # Extract embeddings using the encoder
    model.eval()
    with torch.no_grad():
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
    last_hidden_state = encoder_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
    
    # Mean pooling (ignoring pad tokens)
    # Expand attention mask to match hidden state dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    embeddings = sum_embeddings / sum_mask
    
    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    # Return both PyTorch Tensor and Numpy Array
    return embeddings, embeddings.cpu().numpy()
