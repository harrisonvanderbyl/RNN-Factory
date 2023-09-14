from torch import nn
import torch

from .ShortMem import WaveNet_Mem
from .Cum import CumProd, CumMax

class MatForward(nn.Module):
    def __init__(self, args, layer_id):
        nn.Module.__init__(self)
        self.args = args
        self.layer_id = layer_id
        
        self.complexsize = args.n_embd
        self.short = WaveNet_Mem(args, layer_id, undialated=True)
        self.key = nn.Linear(args.n_embd,self.complexsize*2, bias=False, dtype=torch.bfloat16)
        self.cumprod = CumProd(torch.complex(torch.ones(args.micro_bsz, 1, self.complexsize), torch.zeros(args.micro_bsz, 1, self.complexsize)))
        self.cummax = CumMax()
        self.activation = nn.Linear(self.complexsize*2, args.n_embd, bias=False, dtype=torch.bfloat16)
           
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).float()
        
       
        complexval = torch.view_as_complex(k.reshape(B, T, self.complexsize,2))
        scale = self.cummax(torch.abs(complexval))
        complexval2 = complexval / scale
        kv = self.cumprod(complexval2)
        out = self.activation(torch.view_as_real(kv).reshape(B, T, self.complexsize*2))  * self.short(x* scale)
       
        return out 
    