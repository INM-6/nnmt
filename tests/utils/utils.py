from inspect import signature


def get_required_keys(func, all_keys):
    """Checks arguments of func and returns corresponding parameters."""
    arg_keys = list(signature(func).parameters)
    required_keys = [key for key in all_keys if key in arg_keys]
    return required_keys


def get_required_params(func, all_params):
    """Checks arguments of func and returns corresponding parameters."""
    required_keys = list(signature(func).parameters)
    required_params = {k: v for k, v in all_params.items()
                       if k in required_keys}
    return required_params
