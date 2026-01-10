"""
Config Loader - Caricamento file YAML di configurazione.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carica un file di configurazione YAML.
    
    Args:
        config_path: Path al file YAML
        
    Returns:
        Dizionario con la configurazione
        
    Raises:
        FileNotFoundError: Se il file non esiste
        yaml.YAMLError: Se il file non è un YAML valido
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
    
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Errore nel parsing del file YAML {config_path}: {e}")
    
    return config


def load_all_configs(config_dir: str = "config") -> Dict[str, Dict[str, Any]]:
    """
    Carica tutti i file di configurazione dalla directory config/.
    
    Args:
        config_dir: Directory contenente i file di configurazione
        
    Returns:
        Dizionario con tutte le configurazioni:
        {
            'vehicle_model': {...},
            'detection_params': {...},
            'camera_config': {...}
        }
    """
    config_path = Path(config_dir)
    
    configs = {}
    config_files = {
        'vehicle_model': 'vehicle_model.yaml',
        'detection_params': 'detection_params.yaml',
        'camera_config': 'camera_config.yaml'
    }
    
    for key, filename in config_files.items():
        file_path = config_path / filename
        if file_path.exists():
            configs[key] = load_config(str(file_path))
        else:
            print(f"Warning: File di configurazione {filename} non trovato")
            configs[key] = {}
    
    return configs


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Ottiene un valore annidato da un dizionario usando una path con dot notation.
    
    Args:
        config: Dizionario di configurazione
        key_path: Path con dot notation (es: "vehicle.dimensions.length")
        default: Valore di default se la chiave non esiste
        
    Returns:
        Valore trovato o default
        
    Example:
        >>> config = {'vehicle': {'dimensions': {'length': 5.0}}}
        >>> get_nested_value(config, 'vehicle.dimensions.length')
        5.0
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value