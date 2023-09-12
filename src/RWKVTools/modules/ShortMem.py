# Short Memory
from .TimeShift import TimeShift
import torch
from torch import nn

class Short_Mem(nn.Module):
    def __init__(self, args, shiftAmount=1):
        super().__init__()
        self.time_shift1 = TimeShift(args.n_embd, shiftAmount=shiftAmount, batch=args.micro_bsz)
        self.activation = nn.Sequential(
            nn.Linear(args.n_embd*2, args.n_embd, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        xv = self.activation(torch.cat([self.time_shift1(x),x], dim=-1))
        return  xv

