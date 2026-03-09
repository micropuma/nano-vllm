import torch
import torch.nn as nn

from typing import Protocol

class LayerFn(Protocol):
    def __call__(self) -> nn.Module: ...

def make_layers(
    num_hidden_layers : int,
    layer_func: LayerFn,

) -> nn.Module:
    layers = [layer_func() for _ in range(num_hidden_layers)]
    return layers