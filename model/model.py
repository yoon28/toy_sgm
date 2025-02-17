import torch.nn as nn
import torch

class DenoiseBlock(nn.Module):
    def __init__(self, nunits):
        super(DenoiseBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)
        
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x

class DenoiseModel(nn.Module):
    def __init__(self, nfeatures: int, nblocks: int = 2, nunits: int = 64, is_score=False):
        super(DenoiseModel, self).__init__()
        self.is_score = is_score
        if is_score:
            self.inblock = nn.Linear(nfeatures, nunits)
        else:
            self.inblock = nn.Linear(nfeatures + 1, nunits)
        self.midblocks = nn.ModuleList([DenoiseBlock(nunits) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        if t is None and self.is_score:
            val = x
        else:
            val = torch.hstack([x, t]).type(torch.float32)  # Add t to inputs
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val)
        return val
