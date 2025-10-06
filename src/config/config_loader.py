import yaml

class ConfigLoader:
    """Charge la configuration depuis un fichier YAML."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_config(self, key: str):
        """Retourne une valeur de configuration."""
        return self.config[key]

    def get(self):
        '''Return all the configuration parameters'''
        return self.config

