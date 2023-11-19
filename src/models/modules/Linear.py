import torch
from torch.nn import Linear

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

from torch.utils.cpp_extension import load
load(name="wkv5", sources=["./src/models/modules/cuda/cpuonly.cpp"],
                            verbose=True, extra_cflags=["-O3", "-march=native", "-fopenmp"  ,"-flto",  "-fopenmp", "-funroll-loops", "-D_GLIBCXX_PARALLEL"])


class Quantized(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = None


    def forward(self, x):
        # replicate nn.linear
        out = torch.zeros(x.shape[0] , x.shape[1], self.qweight.shape[0])        
        torch.ops.wkv5.matmul(self.qweight, self.qrange, self.qmin, x, out, x.shape[0], self.qweight.shape[1], x.shape[1], self.qweight.shape[0])
       
        return out
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = list(state_dict.keys())[0]

        weight = state_dict[key].t().contiguous()
        self.qweight = (torch.zeros(weight.shape[1],weight.shape[0]).to(torch.uint8))
        self.qrange = (torch.zeros(weight.shape[1],16))
        self.qmin = (torch.zeros(weight.shape[1],16))
        torch.ops.wkv5.quantize_cpu(state_dict[key].float().contiguous(), self.qrange, self.qmin,self.qweight, weight.shape[1], weight.shape[0])
    

