# Drug Journey

Welcome to the **Drug Journey** repository. This is a master repository designed to house various models and tools related to drug discovery and chemical language models. 

## Repository Structure

The project follows a monorepo-style structure where each machine learning model or specific toolkit has its own isolated subfolder. The virtual environment and high-level requirements are shared across the repository in the root folder.

### Current Models:
- **`nvidia_megamolbart/`**: A standalone PyTorch pipeline for NVIDIA's MegaMolBART model, allowing for drug discovery embeddings, latent interpolations, and SMILES generations. Please see the [Nvidia MegaMolBART README](nvidia_megamolbart/README.md) for specific usage, setup, and details.
- **`chemberta/`**: A pipeline for using DeepChem's ChemBERTa model (transformer for SMILES) from Hugging Face for Masked Language Modeling and embedding extraction. Please see the [ChemBERTa README](chemberta/README.md).

## Setup
1. Create and activate a virtual environment (`venv`) at the root of the repository.
2. Install the shared dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to a specific model's subfolder (e.g., `cd nvidia_megamolbart`) and follow its localized instructions.

## Resources
The root folder also contains summaries and resources related to Chemical Language Models:
- `Paper_Summary_Chemical_Language_Models.md`
- `CLM_Resources.md`
- Relevant research papers (PDFs).
