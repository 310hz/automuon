import torch.nn as nn

from automuon import get_muon_and_adam, with_muon, without_muon

FLAG_NAME = "_automuon_flag"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=2, num_heads=2)
        self.norm = nn.LayerNorm(2)
        self.dropout = nn.Dropout()

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = without_muon(nn.Linear(4, 2))
        self.relu = nn.ReLU()

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.ffn = FFN()

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        self.encoder1 = EncoderLayer()
        self.encoder2 = with_muon(EncoderLayer())
        without_muon(self.encoder2)
        self.head1 = nn.Linear(2, 1)
        self.head2 = with_muon(nn.Linear(2, 1))


def test_assign():
    model = Transformer()
    optimizer_muon, optimizer_adam = get_muon_and_adam(
        model.named_parameters(),
        rule=lambda name, param: "head" not in name and "embed" not in name,
        default="muon",
    )
    params_muon = {
        model.encoder1.attention.attention.in_proj_weight,
        model.encoder1.attention.attention.out_proj.weight,
        model.encoder1.ffn.linear1.weight,
        model.encoder2.attention.attention.in_proj_weight,
        model.encoder2.attention.attention.out_proj.weight,
        model.encoder2.ffn.linear1.weight,
        model.head2.weight,
    }
    params_adam = {
        model.embedding.weight,
        model.encoder1.attention.attention.in_proj_bias,
        model.encoder1.attention.attention.out_proj.bias,
        model.encoder1.attention.norm.weight,
        model.encoder1.attention.norm.bias,
        model.encoder1.ffn.linear1.bias,
        model.encoder1.ffn.linear2.bias,
        model.encoder1.ffn.linear2.weight,
        model.encoder2.attention.attention.in_proj_bias,
        model.encoder2.attention.attention.out_proj.bias,
        model.encoder2.attention.norm.weight,
        model.encoder2.attention.norm.bias,
        model.encoder2.ffn.linear1.bias,
        model.encoder2.ffn.linear2.weight,
        model.encoder2.ffn.linear2.bias,
        model.head1.weight,
        model.head1.bias,
        model.head2.bias,
    }
    _check_optimizer(optimizer_muon, optimizer_adam, params_muon, params_adam)


def _check_optimizer(optimizer_muon, optimizer_adam, params_muon, params_adam):
    params_muon_app = optimizer_muon.param_groups[0]["params"]
    params_adam_app = optimizer_adam.param_groups[0]["params"]
    for param in params_muon:
        assert _is_included(param, params_muon_app)
    for param in params_adam:
        assert _is_included(param, params_adam_app)
    for param in params_muon_app:
        assert _is_included(param, params_muon)
    for param in params_adam_app:
        assert _is_included(param, params_adam)


def _is_included(param, params):
    return any(param is p for p in params)
