from .RNN import Model

from torch.utils.cpp_extension import load
import torch
import os
currentdir = os.path.dirname(os.path.realpath(__file__))

class v5cpp(Model):
    def __init__(self, path):
        self.model_name = 'v5-cpu'
        # load torch.cpp with march=native, O3, and -fopenmp
        self.torch_cpp = load(name="torch_cpp", sources=[currentdir + "/rwkv.cpp", currentdir+"/rwkv.cuh/src/cpuops.cpp"], extra_cflags=["-O3", "-openmp", "-march=native", "-I"+currentdir+"/rwkv.cuh/include"], verbose=True)
        self.torch_cpp.init(path)
        self.batches = 1

        super(v5cpp, self).__init__()
        self.eval()

    def forward(self, idx, state=-1, **kwargs):

        if isinstance(idx, list):
            idx = torch.tensor(idx)
        # if idx is int, make tensor
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        
        # if idx is not 3 dim tensor, make it 3 dim
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
            idx = idx.repeat(self.batches, 1)

        output = torch.zeros(idx.shape[0], idx.shape[1], pow(2, 16), dtype=torch.float32, device=idx.device)
        self.torch_cpp.forward_cpu(idx, output)
        return output, state
    
    def resetState(self):
        self.torch_cpp.resetState()

    def getState(self):
        return None
    
    def setState(self, state):
        pass
    
        