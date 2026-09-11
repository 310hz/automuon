import pytest
import torch
import torch.nn as nn

from automuon import with_adam, with_muon, without_adam, without_muon

FLAG_NAME = "_automuon_flag"


def test_arg_types():
    marking_functions = (
        with_muon,
        without_muon,
        with_adam,
        without_adam,
    )
    objects_ok = [nn.Linear(3, 2), nn.Parameter(torch.randn(3, 2))]
    for obj in objects_ok:
        for marking_function in marking_functions:
            marking_function(obj)

    objects_ng = [1, [1, 2, 3], (1, 2, 3), torch.tensor([1.0, 2.0, 3.0])]
    for obj in objects_ng:
        with pytest.raises(TypeError):
            for marking_function in marking_functions:
                marking_function(obj)


def test_symmetry():
    layers = [
        nn.Linear(3, 2),
        nn.Embedding(3, 2),
        nn.BatchNorm1d(3),
        nn.Sequential(nn.Linear(3, 2), nn.ReLU()),
    ]
    for layer in layers:
        for param in layer.parameters():
            flag_with_muon = getattr(
                with_muon(param),
                FLAG_NAME,
            )
            flag_without_adam = getattr(
                without_adam(param),
                FLAG_NAME,
            )
            flag_without_muon = getattr(
                without_muon(param),
                FLAG_NAME,
            )
            flag_with_adam = getattr(
                with_adam(param),
                FLAG_NAME,
            )
            assert flag_with_muon == flag_without_adam
            assert flag_without_muon == flag_with_adam
