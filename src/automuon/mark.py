from typing import TypeVar

import torch.nn as nn


_Markable = nn.Module | nn.Parameter
_MarkableT = TypeVar("_MarkableT", bound=_Markable)

def with_muon(param: _MarkableT) -> _MarkableT:
    _mark(param, True)
    return param

def without_muon(param: _MarkableT) -> _MarkableT:
    _mark(param, False)
    return param

def with_adam(param: _MarkableT) -> _MarkableT:
    _mark(param, False)
    return param

def without_adam(param: _MarkableT) -> _MarkableT:
    _mark(param, True)
    return param


def _mark(param: _Markable, flag: bool) -> None:
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
