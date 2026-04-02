import yaml
import os
from semantixel.core.config import SemantixelConfig

def load_config(file_path: str) -> dict:
    """
    Load the YAML configuration file and return as a dictionary.
    Uses the Pydantic model to ensure valid defaults.
    """
    if not os.path.exists(file_path):
        return SemantixelConfig().model_dump()
        
    with open(file_path, "r") as file:
        data = yaml.safe_load(file) or {}
    
    # Validate and return as dict for the GUI to manipulate
    try:
        config_obj = SemantixelConfig(**data)
        return config_obj.model_dump()
    except Exception:
        # If validation fails, return the raw data or defaults
        return data

def save_config(config_dict: dict, file_path: str):
    """
    Save the configuration dictionary to a YAML file after validation.
    """
    # Validate before saving
    validated_config = SemantixelConfig(**config_dict)
    
    with open(file_path, "w") as file:
        yaml.dump(validated_config.model_dump(), file, default_flow_style=False)
