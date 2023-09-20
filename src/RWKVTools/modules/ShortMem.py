# Short Memory
from .TimeShift import TimeShift
import torch
from torch import nn
from .Cum import CumProd, CumMax
class Short_Mem(nn.Module):
    def __init__(self, args, shiftAmount=1):
        super().__init__()
        self.time_shift1 = TimeShift(args.n_embd, shiftAmount=shiftAmount, batch=args.micro_bsz)
        self.activation = nn.Sequential(
            nn.Linear(args.n_embd*2, args.n_embd, bias=False, dtype=torch.bfloat16)
           , nn.ReLU(),
           nn.Sigmoid()
        )

        self.value = nn.Linear(args.n_embd, args.n_embd, bias=False, dtype=torch.bfloat16)

        self.cummax = CumMax()
        self.cumprod = CumProd(torch.complex(torch.ones(args.micro_bsz, 1, args.n_embd//2), torch.zeros(args.micro_bsz, 1, args.n_embd//2)))

        self.recetance = nn.Linear(args.n_embd*2, args.n_embd, bias=False, dtype=torch.bfloat16)
        
        
    def forward(self, x):
        B, T, C = x.size()
        ct = torch.cat([self.time_shift1(x),x], dim=-1)
        xv = self.activation(ct).float() # relu sigmoid : R -> (0.5-1.0)

        complexval = torch.view_as_complex(xv.reshape(B, T, -1, 2)) # Transform to complex number. Confined to first quadrant < 90 degrees

        complexval = complexval / self.cummax(torch.abs(complexval)) # Normalize to unit circle, using cumulative max abs to minimize data loss
        
        kv = self.cumprod(complexval) # Cumulative product, in complex space, this corresponds to a cumulative rotation
        
        kv = torch.view_as_real(kv).reshape(B, T, -1) # Transform back to real space

        return  self.value(kv) * self.recetance(ct).relu().sigmoid().pow(4) # relu sigmoid pow(4) acts as an activation, keeping values between 0.125-1.0, preventing exploding values 

class WaveNet_Mem(Short_Mem):
    def __init__(self, args, layer_id, modulo=12, undialated=False, cap=12):
        if undialated:
            super().__init__(args, shiftAmount=1)
        else:
            super().__init__(args, shiftAmount=2**((layer_id%modulo) if layer_id < cap else 0))