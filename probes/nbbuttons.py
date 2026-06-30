import torch
from torch import nn


class ButtonNumberProbe(nn.Module):
    def __init__(self, d_model: int):
        self.probe = nn.Linear(d_model, 1)

    def forward(self, input):
        """
        - input: [B, sum_l(Hl * Wl), embed_dim]
        
        Input is the encoder memory.
        """
        # [B, sum_l(Hl * Wl), embed_dim] -> [B, sum_l(Hl * Wl), 1]
        result = self.probe(input)
        # now we need to sum it to get a meaningful interpretation
        # [B, sum_l(Hl * Wl), 1] -> [B, 1]
        result = result.sum(dim=1)
        return result
    

def train(model):
    pass