from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from automuon import get_muon_and_adam, without_muon


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = without_muon(nn.Linear(2, 2))
        self.linear3 = nn.Linear(2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def test_train():
    model1 = Model()
    model2 = deepcopy(model1)

    optimizer_muon1, optimizer_adam1 = get_muon_and_adam(
        model1.parameters(),
        muon_args={"lr": 0.01, "weight_decay": 0.03, "momentum": 0.9},
        adam_args={"lr": 0.01, "weight_decay": 0.03, "betas": (0.9, 0.95)},
    )

    params_muon = []
    params_adam = []

    for p in model2.linear1.parameters():
        if p.ndim == 2:
            params_muon.append(p)
        else:
            params_adam.append(p)
    for p in model2.linear2.parameters():
        params_adam.append(p)
    for p in model2.linear3.parameters():
        if p.ndim == 2:
            params_muon.append(p)
        else:
            params_adam.append(p)

    optimizer_muon2 = optim.Muon(params_muon, lr=0.01, weight_decay=0.03, momentum=0.9)
    optimizer_adam2 = optim.AdamW(params_adam, lr=0.01, weight_decay=0.03, betas=(0.9, 0.95))

    for _ in range(10):
        x = torch.randn(4, 2)

        optimizer_muon1.zero_grad()
        optimizer_adam1.zero_grad()
        optimizer_muon2.zero_grad()
        optimizer_adam2.zero_grad()

        loss1 = model1(x).mean()
        loss2 = model2(x).mean()

        loss1.backward()
        loss2.backward()

        optimizer_muon1.step()
        optimizer_adam1.step()
        optimizer_muon2.step()
        optimizer_adam2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
