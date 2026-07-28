import torch.nn as nn


def _mark(params, flag):
    original = params
    if isinstance(params, (nn.Parameter, nn.Module)):
        params = [params]
    for param in params:
        if isinstance(param, nn.Module):
            for p in param.parameters():
                setattr(p, "_automuon_flag", flag)
        else:
            setattr(param, "_automuon_flag", flag)
    return original


def with_muon(params):
    return _mark(params, True)

def without_muon(params):
    return _mark(params, False)

def with_adam(params):
    return _mark(params, True)

def without_adam(params):
    return _mark(params, False)
