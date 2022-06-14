import inspect
import os
import nnmt.input_output as io
from nnmt.input_output import load_val_unit_dict
from nnmt.utils import (
    _strip_units,
    _to_si_units,
    )


def get_required_params(func, all_params):
    """Checks arguments of func and returns corresponding parameters."""
    required_keys = list(inspect.signature(func).parameters)
    required_params = {k: v for k, v in all_params.items()
                       if k in required_keys}
    return required_params


def extract_required_params(func, regime_params):
    extracted = [get_required_params(func, params)
                 for params in regime_params]
    return extracted


def create_and_save_fixtures(func, regime_params, regimes, file):
    results = {}
    regime_params = extract_required_params(func,
                                            regime_params)
    for regime, params in zip(regimes, regime_params):
        output = func(**params)
        results[regime] = {
            'params': params,
            'output': output
            }
    io.save_h5(file, results, overwrite_dataset=True)


def load_params_and_regimes(config_path):
    param_files = os.listdir(config_path)
    regimes = [file.replace('.yaml', '').replace('.h5', '')
               for file in param_files]
    regime_params = [load_val_unit_dict(config_path + file)
                     for file in param_files]
    for dict in regime_params:
        _to_si_units(dict)
        _strip_units(dict)
    return regime_params, regimes
