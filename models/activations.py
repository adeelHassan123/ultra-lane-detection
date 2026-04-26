import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'. Choose from: relu, leaky_relu, elu, gelu.")
