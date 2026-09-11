from collections.abc import Callable, Iterable
from typing import Callable, TypeAlias

import torch


Param: TypeAlias = torch.nn.Parameter
NamedParam: TypeAlias = tuple[str, Param]
Params: TypeAlias = Iterable[Param] | Iterable[NamedParam]
Rule: TypeAlias = Callable[[str, Param], bool]


def get_muon_and_adam(
    params: Params,
    muon_args: dict | None = None,
    adam_args: dict | None = None,
    rule: Rule | None = None,
    default: str = "muon",
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:

    muon_params = []
    adam_params = []
    muon_args = muon_args or {}
    adam_args = adam_args or {}

    for param in params:
        if with_muon(param, rule, default):
            muon_params.append(param)
        else:
            adam_params.append(param)

    optimizer_muon = torch.optim.Muon(muon_params, **muon_args)
    optimizer_adam = torch.optim.AdamW(adam_params, **adam_args)
    return optimizer_muon, optimizer_adam


def with_muon(param, rule, default):
    if rule:
        name, param = _check_named_param(param)
    else:
        param = _check_param(param)

    match default:
        case "muon":
            flag = True
        case "adam":
            flag = False
        case _:
            raise ValueError(f"Invalid default: {default}")

    if param.ndim != 2:
        flag = False
    elif hasattr(param, "_automuon_flag"):
        flag = param._automuon_flag
    elif rule:
        flag = not rule(name, param)

    return flag


def _check_named_param(param):
    if isinstance(param, tuple) and len(param) == 2:
        name, param = param
        if isinstance(name, str) and isinstance(param, torch.nn.Parameter):
            return name, param
    raise TypeError(
        "When a rule is provided, each item yielded by the params "
        "iterable must be a (name: str, param: torch.nn.Parameter) "
        "tuple, as returned by torch.nn.Module.named_parameters()."
    )


def _check_param(param):
    if isinstance(param, torch.nn.Parameter):
        return param
    raise TypeError(
        "When no rule is provided, each item yielded by the params "
        "iterable must be a torch.nn.Parameter."
    )
