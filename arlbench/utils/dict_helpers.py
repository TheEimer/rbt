from omegaconf import DictConfig, OmegaConf
from ConfigSpace import Configuration
import numpy as np


def numpy_to_python_dtypes(dictionary: dict) -> dict:
    """Converts numpy dtypes to python dtypes."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = numpy_to_python_dtypes(value)
        elif isinstance(value, np.bool_):
            dictionary[key] = bool(value)
        elif isinstance(value, np.integer):
            dictionary[key] = int(value)
        elif isinstance(value, np.floating):
            dictionary[key] = float(value)
    return dictionary


def to_dict(cfg: DictConfig | Configuration) -> dict:
    """Converts a DictConfig or Configuration object to a dictionary."""
    if isinstance(cfg, Configuration):
        config = dict(cfg)
        config = numpy_to_python_dtypes(config)
        return config
    elif isinstance(cfg, DictConfig):
        config = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(config, dict)

        config = numpy_to_python_dtypes(config)
        return config

