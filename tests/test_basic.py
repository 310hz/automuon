import torch
import torch.nn as nn

from automuon import with_muon, without_muon, with_adam, without_adam
from automuon import get_muon_and_adam


FLAG_NAME = "_automuon_flag"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = with_muon(nn.Linear(3, 2))
        self.fc3 = without_muon(nn.Linear(3, 2))
        self.fc4 = with_adam(nn.Linear(3, 2))
        self.fc5 = without_adam(nn.Linear(3, 2))
        self.param11 = nn.Parameter(torch.randn(3, 2))
        self.param12 = nn.Parameter(torch.randn(3))
        self.param21 = with_muon(nn.Parameter(torch.randn(3, 2)))
        self.param22 = with_muon(nn.Parameter(torch.randn(3)))
        self.param31 = without_muon(nn.Parameter(torch.randn(3, 2)))
        self.param32 = without_muon(nn.Parameter(torch.randn(3)))
        self.param41 = with_adam(nn.Parameter(torch.randn(3, 2)))
        self.param42 = with_adam(nn.Parameter(torch.randn(3)))
        self.param51 = without_adam(nn.Parameter(torch.randn(3, 2)))
        self.param52 = without_adam(nn.Parameter(torch.randn(3)))


def test_marking():
    model = Model()
    assert getattr(model.fc1, FLAG_NAME, None) is None
    assert getattr(model.fc2, FLAG_NAME, None) is None
    assert getattr(model.fc3, FLAG_NAME, None) is None
    assert getattr(model.fc4, FLAG_NAME, None) is None
    assert getattr(model.fc5, FLAG_NAME, None) is None
    assert getattr(model.fc1.weight, FLAG_NAME, None) is None
    assert getattr(model.fc2.weight, FLAG_NAME, None) is True
    assert getattr(model.fc3.weight, FLAG_NAME, None) is False
    assert getattr(model.fc4.weight, FLAG_NAME, None) is False
    assert getattr(model.fc5.weight, FLAG_NAME, None) is True
    assert getattr(model.fc1.bias, FLAG_NAME, None) is None
    assert getattr(model.fc2.bias, FLAG_NAME, None) is True
    assert getattr(model.fc3.bias, FLAG_NAME, None) is False
    assert getattr(model.fc4.bias, FLAG_NAME, None) is False
    assert getattr(model.fc5.bias, FLAG_NAME, None) is True
    assert getattr(model.param11, FLAG_NAME, None) is None
    assert getattr(model.param12, FLAG_NAME, None) is None
    assert getattr(model.param21, FLAG_NAME, None) is True
    assert getattr(model.param22, FLAG_NAME, None) is True
    assert getattr(model.param31, FLAG_NAME, None) is False
    assert getattr(model.param32, FLAG_NAME, None) is False
    assert getattr(model.param41, FLAG_NAME, None) is False
    assert getattr(model.param42, FLAG_NAME, None) is False
    assert getattr(model.param51, FLAG_NAME, None) is True
    assert getattr(model.param52, FLAG_NAME, None) is True


def test_optimizer_default_muon():
    model = Model()
    optimizer_muon, optimizer_adam = get_muon_and_adam(model.parameters())
    params_muon = {
        model.fc1.weight, model.param11,
        model.fc2.weight, model.param21,
        model.fc5.weight, model.param51,
    }
    params_adam = {
        model.fc1.bias, model.param12,
        model.fc2.bias, model.param22,
        model.fc3.weight, model.fc3.bias, model.param31, model.param32,
        model.fc4.weight, model.fc4.bias, model.param41, model.param42,
        model.fc5.bias, model.param52,
    }
    _check_optimizer(optimizer_muon, optimizer_adam, params_muon, params_adam)

    optimizer_muon, optimizer_adam = get_muon_and_adam(model.parameters(), default="muon")
    _check_optimizer(optimizer_muon, optimizer_adam, params_muon, params_adam)


def test_optimizer_default_adam():
    model = Model()
    optimizer_muon, optimizer_adam = get_muon_and_adam(model.parameters(), default="adam")
    params_muon = {
        model.fc2.weight, model.param21,
        model.fc5.weight, model.param51,
    }
    params_adam = {
        model.fc1.weight, model.fc1.bias, model.param11, model.param12,
        model.fc2.bias, model.param22,
        model.fc3.weight, model.fc3.bias, model.param31, model.param32,
        model.fc4.weight, model.fc4.bias, model.param41, model.param42,
        model.fc5.bias, model.param52,
    }
    _check_optimizer(optimizer_muon, optimizer_adam, params_muon, params_adam)


def _check_optimizer(optimizer_muon, optimizer_adam, params_muon, params_adam):
    assert isinstance(optimizer_muon, torch.optim.Muon)
    assert isinstance(optimizer_adam, torch.optim.Adam)
    assert len(optimizer_muon.param_groups) == 1
    assert len(optimizer_adam.param_groups) == 1
    assert set(optimizer_muon.param_groups[0]['params']) == params_muon
    assert set(optimizer_adam.param_groups[0]['params']) == params_adam
