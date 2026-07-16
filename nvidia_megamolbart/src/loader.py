import os
import torch
import logging
from transformers import BartForConditionalGeneration
from typing import Tuple

from .config import get_megamolbart_config, map_nemo_to_huggingface
from .tokenizer import MegaMolBARTTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_dir: str) -> Tuple[BartForConditionalGeneration, MegaMolBARTTokenizer]:
    """
    Loads the MegaMolBART PyTorch checkpoint and tokenizer natively.
    Bypasses the NeMo toolkit by manually mapping state dict keys to a 
    HuggingFace BartForConditionalGeneration model.
    """
    weights_path = os.path.join(model_dir, "model_weights.ckpt")
    vocab_path = os.path.join(model_dir, "36b36f49c3e64962a7b54f1a1ba2b580_megamolbart.vocab")
    regex_path = os.path.join(model_dir, "111b90cc2819425382967ab999101096_megamolbart.model")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # 1. Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading MegaMolBART on device: {device}")

    # 2. Load Tokenizer
    logger.info("Initializing Tokenizer...")
    tokenizer = MegaMolBARTTokenizer(vocab_path=vocab_path, regex_path=regex_path)

    # 3. Create Model Configuration
    logger.info("Initializing Model Architecture...")
    config = get_megamolbart_config()
    config.vocab_size = tokenizer.vocab_size
    model = BartForConditionalGeneration(config)

    # 4. Load Raw PyTorch Checkpoint
    logger.info("Loading PyTorch State Dict...")
    # Map location 'cpu' avoids memory spikes before mapping is complete
    raw_ckpt = torch.load(weights_path, map_location="cpu")
    
    # NeMo checkpoints often nest the state dict inside a 'state_dict' key
    if "state_dict" in raw_ckpt:
        nemo_state_dict = raw_ckpt["state_dict"]
    else:
        nemo_state_dict = raw_ckpt

    # 5. Translate Keys
    logger.info("Translating NeMo State Dict to HuggingFace format...")
    hf_state_dict = map_nemo_to_huggingface(nemo_state_dict, tokenizer.vocab_size)

    # 6. Load Weights into Model
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    
    # Filter out acceptable missing keys (like tie weights or missing causal masks that are auto-generated)
    acceptable_missing = [m for m in missing if "embed_positions" not in m]
    if len(acceptable_missing) > 0:
        logger.warning(f"Missing keys during load: {acceptable_missing[:5]} ...")
    if len(unexpected) > 0:
        logger.warning(f"Unexpected keys during load: {unexpected[:5]} ...")

    model.to(device)
    model.eval()

    # 7. Print Summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("-" * 40)
    logger.info("MegaMolBART Successfully Loaded!")
    logger.info(f"Architecture : 6 Layers, 512 Hidden Dim, 8 Heads")
    logger.info(f"Total Params : {total_params:,}")
    logger.info(f"Device       : {device.type.upper()}")
    logger.info("-" * 40)

    return model, tokenizer
