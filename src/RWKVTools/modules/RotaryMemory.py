from torch import nn
import torch

from .Cum import CumProd, CumMax, CumAvg

class RotaryMemory(nn.Module):
    def __init__(self, width, height, **kwargs) -> None:
        super().__init__()
        
        self.cumprod = CumProd()
        self.process = nn.Linear(width//2,width//2, bias=False, dtype=torch.bfloat16)
        self.width = width
        self.reshape = height != width
        

    def forward(self, x):
        B, T, C = x.size()
        complexval = torch.view_as_complex(x.float().relu().sigmoid().pow(4).reshape(B, T,-1 ,2))
        scale = complexval.abs().add(1e-8)
        complexval2 = complexval/scale
        kv = self.cumprod(complexval2)
        
        kv = kv * self.process(scale.to(self.process.weight.dtype))
        if self.reshape:
            out = kv.real 
        else:
            out = torch.view_as_real(kv).reshape(B, T, -1) 
        
        return out 



class MatForward(nn.Module):
    def __init__(self, args, layer_id):
        nn.Module.__init__(self)
        self.args = args
        self.layer_id = layer_id
        
        self.complexsize = args.n_embd
        self.key = nn.Linear(args.n_embd,self.complexsize*2, bias=False, dtype=torch.bfloat16)
        self.rotary = RotaryMemory(self.complexsize*2,self.complexsize*2)
        self.activation = nn.Linear(self.complexsize*2, args.n_embd, bias=False, dtype=torch.bfloat16)
           
    def forward(self, x):
        B, T, C = x.size()
        # print(B, T, C)
        k = self.key(x).float()
        
        k = self.rotary(k)
       
        out = self.activation(k.to(self.activation.weight.dtype)).sigmoid()
       
        return out 
    