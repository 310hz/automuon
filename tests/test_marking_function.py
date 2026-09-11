import pytest
import torch
import torch.nn as nn

from automuon import with_muon, without_muon, with_adam, without_adam


def test_arg_types():
    objects_ok = [nn.Linear(3, 2), nn.Parameter(torch.randn(3, 2))]
    for obj in objects_ok:
        with_muon(obj)
        without_muon(obj)
        with_adam(obj)
        without_adam(obj)

    objects_ng = [1, [1, 2, 3], (1, 2, 3), torch.tensor([1., 2., 3.])]
    for obj in objects_ng:
        with pytest.raises(TypeError):
            with_muon(obj)
            without_muon(obj)
            with_adam(obj)
            without_adam(obj)


def test_symmetry():
    layers = [
        nn.Linear(3, 2),
        nn.Embedding(3, 2),
        nn.BatchNorm1d(3),
        nn.Sequential(nn.Linear(3, 2), nn.ReLU()),
    ]
    for layer in layers:
        for param in layer.parameters():
            assert with_muon(param)._automuon_flag == without_adam(param)._automuon_flag
            assert without_muon(param)._automuon_flag == with_adam(param)._automuon_flag

