from typing import TypeAlias, TypeVar

import torch.nn as nn


INF = float("inf")

_Markable: TypeAlias = nn.Module | nn.Parameter
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
        for n, p in param.named_parameters():
            level = n.count(".")
            if getattr(p, "_automuon_flag_level", INF) < level:
                continue
            setattr(p, "_automuon_flag", flag)
            setattr(p, "_automuon_flag_level", level)
    elif isinstance(param, nn.Parameter):
        setattr(param, "_automuon_flag", flag)
    else:
        raise TypeError(
            "Expected the input to a marking function to be an "
            "instance of torch.nn.Parameter or torch.nn.Module, but "
            f"got '{type(param)}'."
        )
