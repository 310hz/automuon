import torch.nn as nn


def with_muon(params):
    _mark(params, True)
    return params

def without_muon(params):
    _mark(params, False)
    return params

def with_adam(params):
    _mark(params, False)
    return params

def without_adam(params):
    _mark(params, True)
    return params


def _mark(params, flag):
    if isinstance(params, (nn.Parameter, nn.Module)):
        params = [params]
    for param in params:
        if isinstance(param, nn.Module):
            for p in param.parameters():
                setattr(p, "_automuon_flag", flag)
        elif isinstance(param, nn.Parameter):
            setattr(param, "_automuon_flag", flag)
        else:
            raise TypeError(
                "Expected the input to a marking function to be an "
                "instance of torch.nn.Parameter or torch.nn.Module, "
                f"but got '{type(param)}'."
            )
