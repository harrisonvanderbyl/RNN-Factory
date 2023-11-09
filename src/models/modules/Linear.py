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


class Quantized(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if "range" in ", ".join(state_dict.keys()):
            print("Loading quantized matrix...")
            rangekey = [x for x in state_dict.keys() if "range" in x][0]
            offsetkey = [x for x in state_dict.keys() if "offset" in x][0]
            weightkey = [x for x in state_dict.keys() if "weight" in x][0]
            self.range = state_dict[rangekey].float().cuda()
            self.offset = state_dict[offsetkey].float().cuda()
            self.weight = state_dict[weightkey].t().cuda()
            self.M = self.weight.shape[1]
            self.N = self.weight.shape[0]
        else:
            key = list(state_dict.keys())[0]
            self.weight, self.range, self.offset = self.chunkQuantizeMatrix(state_dict[key])
            self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
            self.range = torch.nn.Parameter(self.range)
            self.offset = torch.nn.Parameter(self.offset)
            self.M = self.weight.shape[0]
            self.N = self.weight.shape[1]

    def chunkQuantizeMatrix(self, x):

        print("Quantizing matrix...")
        
        xx = self.QuantizeMatrix(x.t(), 0)
        toset = xx[0].t().to(dtype=torch.uint8, device="cuda")
        mrange = xx[1].float().to(device="cuda")
        offset = xx[2].float().to(device="cuda")
        print("Memory Used: ", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
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

        out = torch.zeros(B, T, self.M, device="cuda").float()

        # int64_t N, int64_t M, torch::Tensor &x, torch::Tensor &w, torch::Tensor &y, torch::Tensor &r, torch::Tensor &o, int64_t offset, int64_t tokenlength
        
        torch.ops.wkv5.mm8_one(self.N, self.M, y.float(), self.weight, out, self.range, self.offset, 0, T)


        
        return out.to(dtype=y.dtype)
