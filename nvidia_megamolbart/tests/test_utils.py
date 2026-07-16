import sys
sys.path.append(".")
from src.utils import is_valid_smiles, filter_valid_smiles, get_molecular_properties

if __name__ == "__main__":
    print("Testing RDKit Validation...")
    
    # Valid SMILES
    ethanol = "CCO"
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Invalid SMILES (syntactically invalid or chemically impossible)
    invalid1 = "C1CC" # Unclosed ring
    invalid2 = "CC(C)(C)(C)C" # Carbon with 5 bonds
    
    smiles_list = [ethanol, aspirin, invalid1, invalid2]
    
    valid_list = filter_valid_smiles(smiles_list)
    
    print(f"Original list: {smiles_list}")
    print(f"Valid list: {valid_list}")
    
    for s in valid_list:
        props = get_molecular_properties(s)
        print(f"Properties for {s}: {props}")
        
    print("Validation test completed!")
