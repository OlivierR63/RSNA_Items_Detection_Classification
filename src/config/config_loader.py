import yaml
from pathlib import Path

class ConfigLoader:
    """Charge la configuration depuis un fichier YAML."""

    def __init__(self, config_path: str) -> None :
        config_dir = Path(config_path).parent
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Résout les chemins relatifs
        for key in ["dicom_root_dir", "output_dir"]:
            if key in self._config:
                self._config[key] = str(config_dir / self._config[key])
        
                for csv_key in self._config["csv_files"]:
                    self._config["csv_files"][csv_key] = str(config_dir / self._config["csv_files"][csv_key])

    
    def get_config(self, key: str):
        """Retourne une valeur de configuration."""
        return self._config[key]


    def get(self) -> dict :
        '''Return all the configuration parameters'''
        return self._config

