# ChemBERTa Pipeline

A standalone implementation to easily utilize the **ChemBERTa** model (specifically `DeepChem/ChemBERTa-77M-MLM`) via the Hugging Face `transformers` library. 

ChemBERTa is a transformer model trained on SMILES strings, making it excellent at understanding molecular structure, extracting latent embeddings, and property prediction.

## Structure
- `src/loader.py`: Initializes the Hugging Face `AutoModelForMaskedLM` and `AutoTokenizer`.
- `src/embeddings.py`: Generates continuous embeddings from SMILES strings.
- `examples/pipeline_demo.py`: A demo script showing embedding extraction and Masked Language Modeling (MLM).
- `tests/`: Basic validation tests.

## Usage
Since this relies directly on Hugging Face's hub, the weights are downloaded automatically on the first run. Just execute the demo!

```bash
python examples/pipeline_demo.py
```
