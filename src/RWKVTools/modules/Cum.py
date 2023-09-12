# ik ik, but theres some cumulative actions that need state management

from .StateModule import StateModule
from torch import nn
import torch


class CumProd(StateModule):
    def __init__(self, state:torch.Tensor,dim=-2):

        nn.Module.__init__(self)
        self.dim = dim
        self.state = state
        self.origState = state.clone()
    def forward(self, x):
        xx = torch.cat([self.state.to(x.device), x], dim=self.dim)
        yy = xx.cumprod(dim=-2)
        self.setState(yy[:,-1:,:])
        return yy[:,:-1,:]
    
    def resetState(self):
        self.setState(self.origState.clone())

class CumSum(StateModule):
    def __init__(self, state:torch.Tensor,dim=-2):
        nn.Module.__init__(self)
        self.state = state
        self.dim = dim
        self.origState = state.clone()
    def forward(self, x):
        xx = torch.cat([self.state.to(x.device), x], dim=self.dim)
        yy = xx.cumsum()
        self.setState(yy[:,-1:,:])
        return yy[:,:-1,:]
    
    def resetState(self):
        self.setState(self.origState.clone())

class CumMax(StateModule):
    def __init__(self,dim=-2):
        nn.Module.__init__(self)
        self.dim = dim
        self.state = None
    def forward(self, x):
        if self.state is None:
            xx = x
            init = True
        else:
            xx = torch.cat([self.state.to(x.device), x], dim=self.dim)
            init = False
        yy, _ = xx.cummax(dim=-2)
        self.setState(yy[:,-1:,:])
        if init:
            return yy
        return yy[:,1:,:]
    
    def resetState(self):
        self.setState(None)

class CumMin(StateModule):
    def __init__(self,dim=-2):
        nn.Module.__init__(self)
        self.dim = dim
        self.state = None
    def forward(self, x):
        if self.state is None:
            xx = x
            init = True
        else:
            xx = torch.cat([self.state.to(x.device), x], dim=self.dim)
            init = False
        yy, _ = xx.cummin(dim=-2)
        self.setState(yy[:,-1:,:])
        if init:
            return yy
        return yy[:,1:,:]
    
    def resetState(self):
        self.setState(None)