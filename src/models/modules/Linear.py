import torch
from torch.nn import Linear
from bitsandbytes import matmul

class InferenceLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = None
    def forward(self, x):
        # replicate nn.linear
        return  x @ self.weight
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = list(state_dict.keys())[0]
        self.weight = torch.nn.Parameter(state_dict[key].t())


class Quantized(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = list(state_dict.keys())[0]
        self.weight, self.range, self.offset = self.chunkQuantizeMatrix(state_dict[key])
        self.range = self.range
        self.offset = self.offset
        self.M = self.weight.shape[0]
        self.N = self.weight.shape[1]

    def chunkQuantizeMatrix(self, x):
        
        xx = self.QuantizeMatrix(x.t(), 0)
        toset = xx[0].t().to(dtype=torch.uint8, device="cuda")
        mrange = xx[1].to(device="cuda")
        offset = xx[2].to(device="cuda")
        return toset, mrange, offset
    
    def QuantizeMatrix(self, xx, i):
        width = xx.shape[0]
        start = i * width
        end = start + width
        x = xx[start:end].t()
        rang = 255
        mini = x.min(0)[0].double()
        out = x-mini
        ran = out.max(0)[0].double()/rang
        out = out/ran
        fractmat = out.frac().double()
        fractmat = fractmat.mean(0)
        mini = mini.double() + fractmat*ran.double()
        
        return [out.t(),ran.to(torch.double).unsqueeze(0).unsqueeze(0).clone(), mini.to(torch.double).squeeze().clone()]

    def forward(self, y):
        B, T, C = y.shape
        
        yv = y.reshape(-1,C).mv(self.offset.to(dtype=y.dtype))
        yv = yv.reshape(B,-1,1).float()
        a = (y*self.range.to(y.dtype)).float()

        
        m = a @ self.weight.t().float()
        
    
        yv = yv + m
        
        return yv.to(dtype=y.dtype)
