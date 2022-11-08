import os

import haiku as hk
import numpy as np


def save_params(params, path):
    """
    Save parameters.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.savez(path, **params)


def unroll(x: dict):
    for k in x.keys():
        if isinstance(x[k], dict):
            x[k] = unroll(x[k])
        elif isinstance(x[k], np.ndarray):
            if x[k].dtype == np.object:
                x[k] = x[k].item()
    return x

def load_params(path):
    """
    Load parameters.
    """
    data = dict(np.load(path, allow_pickle=True))
    data = unroll(data)
    return hk.data_structures.to_immutable_dict(data)
