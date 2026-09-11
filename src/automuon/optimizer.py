from collections.abc import Callable, Iterable
from typing import Literal

import torch

from .mark import FLAG_NAME


type Param = torch.nn.Parameter
type NamedParam = tuple[str, Param]
type Params = Iterable[Param] | Iterable[NamedParam]
type Rule = Callable[[str, Param], bool]


def get_muon_and_adam(
    params: Params,
    muon_args: dict | None = None,
    adam_args: dict | None = None,
    rule: Rule | None = None,
    default: Literal["muon", "adam"] = "muon",
) -> tuple[torch.optim.Muon, torch.optim.AdamW]:

    muon_params = []
    adam_params = []
    muon_args = muon_args or {}
    adam_args = adam_args or {}

    for param in params:
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
                raise ValueError(
                    f"Invalid default value: {default}. "
                    "Must be 'muon' or 'adam'."
                )

        if param.ndim != 2:
            flag = False
        elif hasattr(param, FLAG_NAME):
            flag = getattr(param, FLAG_NAME)
        elif rule:
            flag = rule(name, param)

        if flag:
            muon_params.append(param)
        else:
            adam_params.append(param)

    optimizer_muon = torch.optim.Muon(muon_params, **muon_args)
    optimizer_adam = torch.optim.AdamW(adam_params, **adam_args)
    return optimizer_muon, optimizer_adam


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
