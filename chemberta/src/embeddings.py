import torch
import numpy as np
from typing import Union, List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

def get_chemberta_embedding(
    smiles: Union[str, List[str]], 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer,
    normalize: bool = False
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extracts numerical embeddings for one or more SMILES strings using ChemBERTa.
    
    Args:
        smiles (str or List[str]): The input SMILES string(s).
        model (PreTrainedModel): The loaded ChemBERTa model.
        tokenizer (PreTrainedTokenizer): The loaded ChemBERTa tokenizer.
        normalize (bool): Whether to L2 normalize the resulting embeddings.
        
    Returns:
        torch.Tensor: The embeddings as a PyTorch tensor.
        np.ndarray: The embeddings as a NumPy array.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
        
    # Tokenize the input SMILES
    inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        # output_hidden_states=True guarantees we get the hidden states
        outputs = model(**inputs, output_hidden_states=True)
        
    # Get the hidden states from the last layer
    # For AutoModelForMaskedLM, outputs.hidden_states is a tuple of all layers
    last_hidden_state = outputs.hidden_states[-1]
    
    # Perform mean pooling across the sequence length, respecting the attention mask
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    
    embeddings = sum_embeddings / sum_mask
    
    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    return embeddings, embeddings.cpu().numpy()
