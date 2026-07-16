import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chemberta_model(model_name: str = "DeepChem/ChemBERTa-77M-MLM") -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Loads a pre-trained ChemBERTa model and tokenizer from Hugging Face.
    By default, it uses DeepChem's ChemBERTa-77M-MLM.
    """
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"Loading model {model_name}...")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    logger.info("ChemBERTa Successfully Loaded!")
    return model, tokenizer
