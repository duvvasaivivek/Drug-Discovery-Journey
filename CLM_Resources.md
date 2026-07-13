# Chemical Language Models (CLMs) & Resources

This document serves as a directory of available Chemical Language Models, repositories, and online sources for AI-driven drug discovery and molecular design. 

---

## 1. Generative Models (De Novo Design)
Models designed specifically to "dream up" and generate completely novel chemical structures from scratch.

*   **MolGPT**
    *   **Description:** A Transformer-Decoder model (like GPT) trained autoregressively on SMILES strings to generate valid novel molecules.
    *   **Original Code:** [GitHub - devalab/molgpt](https://github.com/devalab/molgpt)
    *   **Pre-trained Weights:** [Hugging Face - jonghyunlee/MolGPT](https://huggingface.co/jonghyunlee/MolGPT_pretrained-by-ZINC15)
*   **MolGPT 2.0**
    *   **Description:** An updated framework building upon the original MolGPT architecture for better conditional generation.
    *   **Original Code:** [GitHub - devalab/MolGPT2.0](https://github.com/devalab/MolGPT2.0)
*   **MegaMolBART (NVIDIA)**
    *   **Description:** A massive generative chemistry model built by NVIDIA (part of BioNeMo) capable of reaction prediction, molecular generation, and property prediction.
    *   **Source:** [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/megamolbart)

## 2. Representation & Property Prediction Models
Models designed to "read" and understand existing molecules to predict their properties (toxicity, solubility, etc.).

*   **ChemBERTa**
    *   **Description:** The chemical equivalent of BERT. It uses Masked Language Modeling to learn deep representations of SMILES strings, making it excellent for predicting chemical properties.
    *   **Original Code / Paper:** [DeepChem / ChemBERTa](https://github.com/deepchem/deepchem)
    *   **Pre-trained Weights:** [Hugging Face - seyonec/ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
*   **MolecularGPT**
    *   **Description:** An instruction-tuned Large Language Model specifically fine-tuned for few-shot molecular property prediction based on textual instructions.
    *   **Original Code:** [GitHub - NYUSHCS/MolecularGPT](https://github.com/NYUSHCS/MolecularGPT)

## 3. Specialized Academic Models (Bajorath Group)
The specialized models mentioned in the *Molecular Informatics* paper summary. While less likely to have "plug-and-play" Hugging Face weights, their methodologies are highly influential.

*   **DeepSARM:** Uses deep generative models for fragment-based design and Structure-Activity Relationship (SAR) exploration.
*   **DeepCubist:** A Transformer-based model specifically for designing complex 3D peptidomimetics and protein-protein interaction inhibitors.
*   **DeepAC:** Focuses specifically on predicting "activity cliffs"—tiny structural changes that cause massive jumps in drug potency.
*   **Motif2Mol:** Translates directly from a biological sequence (a protein's binding site) into a chemical sequence (a drug inhibitor).
    *   *Note: Code for these is typically found in the Supplemental Information attached to their respective academic journal publications, rather than a central GitHub hub.*

## 4. General Hubs & Toolkits
Where to find more models and the software infrastructure to run them.

*   **Hugging Face (Chemistry Tag):** Searching the `chemistry` or `SMILES` tags on Hugging Face yields hundreds of community-trained models.
    *   [Hugging Face Models](https://huggingface.co/models?pipeline_tag=chemistry)
*   **RDKit:** Not an AI model, but the absolutely essential open-source chemoinformatics software used to validate, draw, and manipulate the SMILES strings that these AIs generate.
    *   [RDKit Official Site](https://www.rdkit.org/)
*   **DeepChem:** A massive open-source library specifically for applying deep learning to drug discovery, materials science, and quantum chemistry.
    *   [DeepChem GitHub](https://github.com/deepchem/deepchem)
