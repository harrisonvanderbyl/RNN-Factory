import torch
from torch import nn

from .StateModule import StateModule

class TimeShift(StateModule):
    def __init__(self, dims, shiftAmount=1, batch=1 , *args, **kwargs) -> None:
        super().__init__(batch,shiftAmount, dims)
        self.shift = shiftAmount
        self.batch = batch
        self.dims = dims
       
    def forward(self, x):
 
        xapp = torch.cat([self.getState(x).to(x.device, x.dtype), x], dim=-2)
        self.setState(xapp[:,-self.shift:])
        return xapp[:,:-self.shift,:]
    
    def getState(self,x):
        if self.state is None:
            self.initState(x.shape[0], self.shift, self.dims)
        return self.state.clone()
    
    def resetState(self):
        self.state = None
    