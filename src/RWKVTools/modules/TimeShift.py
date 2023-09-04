import torch
from torch import nn

class TimeShift(nn.Module):
    def __init__(self, dims, shiftAmount=1, batch=1 , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = torch.zeros(batch,shiftAmount, dims)
        self.shift = shiftAmount
    
    
    def forward(self, x):
        tokens = x.shape[-2]
        xapp = torch.cat([self.state.to(x.device, x.dtype), x], dim=-2)
        if self.training:
            pass
        else:
            self.state = xapp[:,-self.shift:,:].clone()
        return xapp[:,:tokens,:]
    
    def setState(self, state):
        self.state = state.clone()

    def getState(self):
        return self.state.clone()