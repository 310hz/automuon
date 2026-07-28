import torch.nn as nn


def _mark(params, flag):
    if isinstance(params, (nn.Parameter, nn.Module)):
        params = [params]
    for param in params:
        if isinstance(param, nn.Module):
            for p in param.parameters():
                setattr(p, "_automuon_flag", flag)
        else:
            setattr(param, "_automuon_flag", flag)


def with_muon(params):
    _mark(params, True)
    return params

def without_muon(params):
    _mark(params, False)
    return params

def with_adam(params):
    _mark(params, True)
    return params

def without_adam(params):
    _mark(params, False)
    return params
