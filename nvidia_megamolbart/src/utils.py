from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional

def is_valid_smiles(smiles: str) -> bool:
    """
    Checks if a SMILES string represents a valid, synthesizable molecule using RDKit.
    Returns False if RDKit cannot parse the string or sanitize it.
    """
    # Disable RDKit C++ warnings in the console to prevent flooding
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        # If it parsed, it's structurally valid according to RDKit sanitization (done by default)
        return True
    except Exception:
        return False

def filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    Takes a list of generated SMILES strings and returns only the valid ones.
    """
    return [s for s in smiles_list if is_valid_smiles(s)]

def get_molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """
    Returns basic chemical properties for a valid SMILES string.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    properties = {
        "MolecularWeight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol)
    }
    
    return properties
