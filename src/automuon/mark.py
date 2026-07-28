import torch.nn as nn


def with_muon(param):
    _mark(param, True)
    return param

def without_muon(param):
    _mark(param, False)
    return param

def with_adam(param):
    _mark(param, False)
    return param

def without_adam(param):
    _mark(param, True)
    return param


def _mark(param, flag):
    if isinstance(param, nn.Module):
        for p in param.parameters():
            setattr(p, "_automuon_flag", flag)
    elif isinstance(param, nn.Parameter):
        setattr(param, "_automuon_flag", flag)
    else:
        raise TypeError(
            "Expected the input to a marking function to be an "
            "instance of torch.nn.Parameter or torch.nn.Module, but "
            f"got '{type(param)}'."
        )
