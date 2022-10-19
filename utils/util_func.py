

def maybe_kwargs(kwargs_to_check, defaults: dict = None):
    kwargs_to_check = {} if kwargs_to_check is None else kwargs_to_check
    if defaults is not None:
        for k, v in defaults.items():
            kwargs_to_check = maybe_default_kwarg(kwargs_to_check, k, v)
    return kwargs_to_check

def maybe_default_kwarg(kwargs_dict, kwarg_to_check, default_value):
    kwargs_dict = maybe_kwargs(kwargs_dict)
    if kwarg_to_check not in kwargs_dict.keys():
        kwargs_dict[kwarg_to_check] = default_value
    return kwargs_dict


def extract_kwarg(kwargs_dict, kwarg_to_check, extract_default, kwarg_default):
    if kwarg_to_check in kwargs_dict.keys():
        extracted = kwargs_dict[kwarg_to_check]
        kwargs_dict[kwarg_to_check] = kwarg_default
    else:
        extracted = extract_default
    return extracted
