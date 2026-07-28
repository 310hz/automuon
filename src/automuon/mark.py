import torch.nn as nn


def use_muon(
    params: nn.Parameter | nn.Module | list[nn.Parameter | nn.Module]
) -> nn.Parameter | nn.Module | list[nn.Parameter | nn.Module]:
    original = params
    if isinstance(params, (nn.Parameter, nn.Module)):
        params = [params]
    for param in params:
        if isinstance(param, nn.Module):
            for p in param.parameters():
                setattr(p, "_automuon_use_muon", True)
        else:
            setattr(param, "_automuon_use_muon", True)
    return original
