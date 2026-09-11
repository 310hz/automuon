# automuon

[![PyPI](https://img.shields.io/pypi/v/automuon)](https://pypi.org/project/automuon/)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?logo=pytorch&logoColor=white)

Automatically split your PyTorch model's parameters between Muon and Adam.

## Installation

Requires Python 3.12+ and PyTorch 2.9+.

```bash
pip install automuon
```

## Quickstart

Mark modules or parameters you want to exclude from Muon with `without_muon`, then pass the model's parameters to `get_muon_and_adam`. It returns two optimizers with their parameters already assigned: Muon (`torch.optim.Muon`) and Adam (`torch.optim.AdamW`).

Parameters that are not 2D always go to Adam, regardless of marks or rules.

```python
import torch.nn as nn
from automuon import without_muon, get_muon_and_adam

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = without_muon(nn.Embedding(...))
        self.transformer_encoder = nn.TransformerEncoder(...)
        self.head = without_muon(nn.Linear(...))

model = Model()
optimizer_muon, optimizer_adam = get_muon_and_adam(
    model.parameters(),
    muon_args={"lr": 1e-3},
    adam_args={"lr": 1e-3, "betas": (0.9, 0.95)},
)

```

You can also use a `rule` to assign parameters without marking them. Pass `model.named_parameters()` so the rule can inspect parameter names:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(...)
        self.transformer_encoder = nn.TransformerEncoder(...)
        self.head = nn.Linear(...)

model = Model()
optimizer_muon, optimizer_adam = get_muon_and_adam(
    model.named_parameters(),
    rule=lambda name, param: ("embed" not in name) and ("head" not in name),
)
```

## API Reference

### Marking Functions

Four marking functions are available:

```python
from automuon import with_muon, without_muon, with_adam, without_adam
```

`with_muon` and `without_adam` are equivalent, as are `without_muon` and `with_adam`. Use whichever reads more naturally in your code.

These functions mark an `nn.Module` or `nn.Parameter` for assignment to Muon or Adam when passed to `get_muon_and_adam`. They return the original object. Marking a module marks all of its current parameters, including those in its submodules.

For unmarked 2D parameters, `rule` applies if provided; otherwise, the `default` argument determines the optimizer (`"muon"` by default).

If a parameter is marked multiple times, the mark from deeper in the module hierarchy takes precedence. At the same depth, the first mark wins.

In this example, `model.layer.fc1.weight` goes to Muon, and all other parameters go to Adam:

```python
class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = with_muon(nn.Linear(...))
        self.fc2 = nn.Linear(...)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = without_muon(Layer())

model = Model()
```

### `get_muon_and_adam`

Splits parameters between `torch.optim.Muon` and `torch.optim.AdamW` and returns the two optimizer instances, in that order.

```python
get_muon_and_adam(
    params: Params,
    muon_args: dict | None = None,
    adam_args: dict | None = None,
    rule: Rule | None = None,
    default: Literal["muon", "adam"] = "muon",
) -> tuple[torch.optim.Muon, torch.optim.AdamW]
```

- `params`: An iterable of `nn.Parameter` objects, such as `model.parameters()`. When `rule` is provided, pass an iterable of `(name, param)` tuples instead, such as `model.named_parameters()`.
- `muon_args`: A dictionary of keyword arguments passed to torch.optim.Muon.
- `adam_args`: A dictionary of keyword arguments passed to torch.optim.Adam.
- `rule`: A callable invoked as `rule(name, param)` for unmarked 2D parameters. Return `True` for Muon or `False` for Adam.
- `default`: The optimizer for parameters not assigned by the conditions below. Either `"muon"` (the default) or `"adam"`.

### Assignment Priority

Parameters are assigned in the following order of precedence:

1. Dimensionality: Parameters that are not 2D always go to Adam.
2. Marks: An explicit mark determines the optimizer.
3. Rule: The result of `rule(name, param)` determines the optimizer.
4. Default: The `default` argument is used as the fallback.
