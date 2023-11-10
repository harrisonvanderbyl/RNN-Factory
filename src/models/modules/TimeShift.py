import torch
from torch import nn

from .StateModule import StateModule

class TimeShift(StateModule):
    def __init__(self, dims, shiftAmount=1, batch=1 , *args, **kwargs) -> None:
        super().__init__(batch,shiftAmount, dims)
        self.shift = shiftAmount
        self.batch = batch
        self.dims = dims
       
    def forward(self, x, state):
 
        xapp = torch.cat([self.getState(state,x), x], dim=-2)
        
        return xapp[:,:-self.shift,:], xapp[:,-self.shift:]
    
    def getState(self,state,x):
        if state is None:
            return torch.zeros(x.shape[0], self.shift, self.dims, device=x.device, dtype=x.dtype)
        return state
    
    def resetState(self):
        self.state = None
    