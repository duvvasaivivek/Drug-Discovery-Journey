# MegaMolBART PyTorch Pipeline

A native, standalone PyTorch pipeline for NVIDIA's MegaMolBART model. 
This project allows you to run drug discovery embeddings, latent interpolations, and SMILES generations **without** requiring NVIDIA's heavy `nemo_toolkit` dependencies (which often fail to compile on Windows).

## Features
- **Standalone PyTorch Loader:** Parses NVIDIA's `.nemo` weights and seamlessly maps them to a HuggingFace `BartForConditionalGeneration` architecture.
- **Custom Tokenizer:** Fully replicates the NVIDIA Regex SMILES tokenization rules.
- **Latent Space Embedding:** Extracts 512-dimensional continuous vectors representing the chemical meaning of molecules.
- **Molecular Interpolation:** Interpolates between the embeddings of two different drugs and uses the Decoder to generate novel hybrid molecules.
- **RDKit Validation:** Filters out chemically impossible outputs and calculates real-world properties (Molecular Weight, LogP, TPSA) for valid discoveries.

## Project Structure
```text
project/
├── models/
│   └── megamolbart/            # (Download the .nemo file and extract it here)
├── src/
│   ├── config.py               # Key mapping and HuggingFace BartConfig
│   ├── loader.py               # Main model loader
│   ├── tokenizer.py            # Regex-based SMILES tokenizer
│   ├── embeddings.py           # Functions to extract numerical vectors
│   ├── generation.py           # Decoder generation and latent interpolation
│   └── utils.py                # RDKit validation and property extraction
├── examples/
│   └── pipeline_demo.py        # End-to-end demo script
└── scripts/
    └── benchmark.py            # Performance measurement script
```

## Setup & Requirements

1. Ensure you have Python 3.10+ installed.
2. Install the necessary lightweight dependencies:
   ```bash
   pip install torch transformers numpy rdkit
   ```
3. Download the MegaMolBART `.nemo` checkpoint from the NVIDIA NGC catalog.
4. Extract the `.nemo` file (it is just a tar archive) into the `models/megamolbart` directory. You should see `model_weights.ckpt`, `model_config.yaml`, and the vocab/model files inside.

## Usage

To run the end-to-end pipeline demonstrating embeddings, latent interpolation, and validation, run:

```bash
python examples/pipeline_demo.py
```

### Performance

Because the model natively maps to HuggingFace BART, it benefits from highly optimized PyTorch attention mechanisms. Even on a standard CPU, extracting embeddings for a batch of molecules takes only a fraction of a second. Check the `scripts/benchmark.py` script for exact performance numbers on your hardware.
