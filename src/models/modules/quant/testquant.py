

import torch
import time
# load ./quant.cpp extension into pytorch

from torch.utils.cpp_extension import load
quant_cpp = load(name="quant_cpp", sources=["./quant.cpp"], verbose=True,
        extra_cflags=["-O3", "-march=native", "-fopenmp"  ,"-flto",  "-fopenmp", "-funroll-loops", "-D_GLIBCXX_PARALLEL"])

class InferenceLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        weight = (torch.rand(4096,2048))
        self.weight = weight
        self.qweight = (torch.zeros(weight.shape[1],weight.shape[0]).to(torch.uint8))
        self.qrange = (torch.zeros(weight.shape[1],16))
        self.qmin = (torch.zeros(weight.shape[1],16))
        quant_cpp.quantize_cpu(self.weight.t().contiguous(), self.qrange, self.qmin,self.qweight, self.weight.shape[1], self.weight.shape[0])
    def forward(self, x):
        # replicate nn.linear
        intime = time.time()
        print((x @ self.weight)[-1,:,-5:])
        outtime = time.time()
        print("Pytorch time: ", outtime-intime)
        out = torch.zeros(x.shape[0] , x.shape[1], self.qweight.shape[0])

        intime = time.time()
        
        quant_cpp.matmul(self.qweight, self.qrange, self.qmin, x, out, x.shape[0], self.qweight.shape[1], x.shape[1], self.qweight.shape[0])
        outtime = time.time()
        print(out[-1,:,-5:])
        print("Quantized time: ", outtime-intime)

        print("Difference: ", ((x @ self.weight) - out ).abs().max())
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = list(state_dict.keys())[0]
        self.weight = torch.nn.Parameter(state_dict[key].t())



quant = InferenceLinear()

x = torch.randn(8,8,4096)
quant(x)