import os
import yaml

def load_configs(config_filepath = 'config.yml'):
    with open(config_filepath, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

