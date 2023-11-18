

import torch
import time
# load ./quant.cpp extension into pytorch

from torch.utils.cpp_extension import load
quant_cpp = load(name="quant_cpp", sources=["./quant.cpp"], verbose=True,
        extra_cflags=["-O3", "-march=native"])

class InferenceLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = (torch.rand(1024,1024))
        self.qweight = (torch.zeros(1024,1024).to(torch.uint8))
        self.qrange = (torch.zeros(1024,16))
        self.qmin = (torch.zeros(1024,16))
        quant_cpp.quantize_cpu(self.weight, self.qrange, self.qmin,self.qweight, 1024,1024)
    def forward(self, x):
        # replicate nn.linear
        intime = time.time()
        print((self.weight @ x).reshape(-1)[-10:])
        outtime = time.time()
        print("Pytorch time: ", outtime-intime)
        intime = time.time()
        out = torch.empty(x.shape[0] , x.shape[1], self.weight.shape[1])

        quant_cpp.matmul(self.qweight, self.qrange, self.qmin, x, out, x.shape[0], 1024, 1024, 1024)
        outtime = time.time()
        print(out.reshape(-1)[-10:])
        print("Quantized time: ", outtime-intime)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = list(state_dict.keys())[0]
        self.weight = torch.nn.Parameter(state_dict[key].t())



quant = InferenceLinear()

x = torch.randn(100,1024,1024)
quant(x)