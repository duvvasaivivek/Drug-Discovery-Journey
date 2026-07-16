import os
import re
from typing import List, Union

class MegaMolBARTTokenizer:
    """
    A custom tokenizer for MegaMolBART that uses the official regex pattern
    and vocabulary file provided in the .nemo checkpoint.
    """
    def __init__(self, vocab_path: str, regex_path: str):
        self.vocab_path = vocab_path
        self.regex_path = regex_path
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        self._load_vocab()
        self._load_regex()
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.mask_token = "<MASK>"
        self.sep_token = "<SEP>"
        self.bos_token = "^"
        self.eos_token = "&"
        
        self.pad_token_id = self.token_to_id.get(self.pad_token, 0)
        self.mask_token_id = self.token_to_id.get(self.mask_token, 4)
        self.bos_token_id = self.token_to_id.get(self.bos_token, 2)
        self.eos_token_id = self.token_to_id.get(self.eos_token, 3)

    def _load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {self.vocab_path}")
            
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            
        for i, token in enumerate(lines):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def _load_regex(self):
        if not os.path.exists(self.regex_path):
            raise FileNotFoundError(f"Regex model file not found at {self.regex_path}")
            
        with open(self.regex_path, "r", encoding="utf-8") as f:
            pattern_str = f.read().strip()
            
        self.pattern = re.compile(pattern_str)

    def tokenize(self, smiles: str) -> List[str]:
        """Splits a SMILES string into tokens using the regex pattern."""
        return self.pattern.findall(smiles)

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """Converts a SMILES string to a list of token IDs."""
        tokens = self.tokenize(smiles)
        
        # Unknown tokens fall back to some id (usually UNK, but not explicitly defined, we skip or use a generic unused)
        # We will just map it if it exists
        token_ids = []
        for t in tokens:
            if t in self.token_to_id:
                token_ids.append(self.token_to_id[t])
            else:
                # Fallback to character level or ignore? In SMILES, valid tokens should match.
                pass 
                
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converts token IDs back to a SMILES string."""
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id, self.token_to_id.get(self.sep_token)]:
                continue
            tokens.append(self.id_to_token.get(tid, ""))
        return "".join(tokens)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
