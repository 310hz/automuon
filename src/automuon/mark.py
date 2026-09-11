import torch.nn as nn


INF = float("inf")

type _Markable = nn.Module | nn.Parameter


def with_muon[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    _mark(param, True)
    return param


def without_muon[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    _mark(param, False)
    return param


def with_adam[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    _mark(param, False)
    return param


def without_adam[MarkableT: _Markable](param: MarkableT) -> MarkableT:
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
