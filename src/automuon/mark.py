import torch.nn as nn

INF = float("inf")
FLAG_NAME = "_automuon_flag"
FLAG_LEVEL_NAME = "_automuon_flag_level"

type _Markable = nn.Module | nn.Parameter


def with_muon[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    """Mark a module or parameter for assignment to Muon.

    Args:
        param: The module or parameter to mark.

    Returns:
        The original module or parameter.
    """
    _mark(param, True)
    return param


def without_muon[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    """Mark a module or parameter for exclusion from Muon.

    Args:
        param: The module or parameter to mark.

    Returns:
        The original module or parameter.
    """
    _mark(param, False)
    return param


def with_adam[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    """Mark a module or parameter for assignment to Adam.

    Args:
        param: The module or parameter to mark.

    Returns:
        The original module or parameter.
    """
    _mark(param, False)
    return param


def without_adam[MarkableT: _Markable](param: MarkableT) -> MarkableT:
    """Mark a module or parameter for exclusion from Adam.

    Args:
        param: The module or parameter to mark.

    Returns:
        The original module or parameter.
    """
    _mark(param, True)
    return param


def _mark(param: _Markable, flag: bool) -> None:
    if isinstance(param, nn.Module):
        for n, p in param.named_parameters():
            level = n.count(".")
            if getattr(p, FLAG_LEVEL_NAME, INF) > level:
                setattr(p, FLAG_NAME, flag)
                setattr(p, FLAG_LEVEL_NAME, level)
    elif isinstance(param, nn.Parameter):
        if getattr(param, FLAG_LEVEL_NAME, INF) > 0:
            setattr(param, FLAG_NAME, flag)
            setattr(param, FLAG_LEVEL_NAME, 0)
    else:
        raise TypeError(
            "Expected the input to a marking function to be an "
            "instance of torch.nn.Parameter or torch.nn.Module, but "
            f"got '{type(param)}'."
        )
