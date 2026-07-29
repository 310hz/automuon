from collections.abc import Callable, Iterable
from typing import Callable, TypeAlias

import torch
import torch.distributed as dist
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


Param: TypeAlias = torch.nn.Parameter
NamedParam: TypeAlias = tuple[str, Param]
Params: TypeAlias = Iterable[Param] | Iterable[NamedParam]
Rule: TypeAlias = Callable[[str, Param], bool]


def get_muon_with_adam(
    params: Params,
    muon_args: dict | None = None,
    adam_args: dict | None = None,
    rule: Rule | None = None,
    default_optim: str = "muon",
    distributed: bool | None = None,
) -> torch.optim.Optimizer:

    if distributed is None:
        distributed = dist.is_available() and dist.is_initialized()

    muon_params = []
    adam_params = []
    for param in params:
        if rule:
            name, param = _check_named_param(param)
        else:
            param = _check_param(param)

        match default_optim:
            case "muon":
                with_muon = True
            case "adam":
                with_muon = False
            case _:
                raise ValueError(f"Invalid default_optim: {default_optim}")

        if param.ndim < 2:
            with_muon = False
        elif hasattr(param, "_automuon_flag"):
            with_muon = param._automuon_flag
        elif rule:
            with_muon = not rule(name, param)

        if with_muon:
            muon_params.append(param)
        else:
            adam_params.append(param)

    muon_args = muon_args or {}
    adam_args = adam_args or {}
    param_groups = [
        dict(params=muon_params, use_muon=True, **muon_args),
        dict(params=adam_params, use_muon=False, **adam_args)
    ]

    if distributed:
        return MuonWithAuxAdam(param_groups)
    else:
        return SingleDeviceMuonWithAuxAdam(param_groups)


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
