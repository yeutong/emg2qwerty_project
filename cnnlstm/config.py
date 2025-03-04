import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        first_line = f.readline()
        if first_line.startswith('# @package'):
            cfg = yaml.safe_load(f)
        else:
            f.seek(0)
            cfg = yaml.safe_load(f)
    
    return cfg
