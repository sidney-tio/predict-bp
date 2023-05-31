import os
import yaml
from easydict import EasyDict


def load_config(config_file='config.yaml'):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, config_file)
    debug = eval(os.environ.get('DEBUG', 'True'))
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cfg = config['dev'] if debug else config['prod']
    return EasyDict(cfg)


config = load_config()