import torch
from torch import nn

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)