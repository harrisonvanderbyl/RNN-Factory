# Short Memory
from .TimeShift import TimeShift
import torch
from torch import nn

class Short_Mem(nn.Module):
    def __init__(self, args, shiftAmount=1, layer=1):
        super().__init__()
        self.args = args
        self.layer = layer
        self.time_shift1 = TimeShift(args.n_embd*layer, shiftAmount=shiftAmount, batch=args.micro_bsz)
        print(args.n_embd*(1+layer))
        self.activation = nn.Sequential(
            nn.Linear(args.n_embd*(2*self.layer), args.n_embd, bias=False),
            nn.ReLU(),
        )
        
    def forward(self, x):
        xv = self.activation(torch.cat([self.time_shift1(x),x], dim=-1))
        return  xv

