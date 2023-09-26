# ik ik, but theres some cumulative actions that need state management

from .StateModule import StateModule
from torch import nn
import torch


class CumProd(StateModule):
    def __init__(self ,dim=-2):

        nn.Module.__init__(self)
        self.dim = dim
        self.state = None
        self.origState = None
    def forward(self, x):
        state = self.getState()
        if state is None:
            if x.dtype == torch.complex64 or x.dtype == torch.complex128:
                state = torch.complex(torch.ones(x.shape[0], 1, x.shape[-1]), torch.zeros(x.shape[0], 1, x.shape[-1]))
            else:
                state = torch.ones(x.shape[0], 1, x.shape[-1])
        xx = torch.cat([state.to(x.device), x], dim=self.dim)
        yy = xx.cumprod(dim=-2)
        self.setState(yy[:,-1:,:])
        return yy[:,1:,:]
    
    def resetState(self):
        self.setState(None)



class CumSum(StateModule):
    def __init__(self ,dim=-2, modulo= None):

        nn.Module.__init__(self)
        self.dim = dim
        self.state = None
        self.origState = None
        self.modulo = modulo
    def forward(self, x):
        state = self.getState()
        if state is None:
            if x.dtype == torch.complex64 or x.dtype == torch.complex128:
                state = torch.complex(torch.zeros(x.shape[0], 1, x.shape[-1]), torch.zeros(x.shape[0], 1, x.shape[-1]))
            else:
                state = torch.ones(x.shape[0], 1, x.shape[-1])
        xx = torch.cat([state.to(x.device), x], dim=self.dim)
        yy = xx.cumsum(dim=-2)
        self.setState(yy[:,-1:,:])
        return yy[:,1:,:]
    
    def resetState(self):
        self.setState(None)

    def getState(self):
        return super().getState() if self.modulo is None or self.state is None else self.state % self.modulo

class CumAvg(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.cumsum = CumSum()
        self.cumcount = CumSum()
    def forward(self, x):
        return self.cumsum(x) / self.cumcount(torch.ones_like(x))

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

### This is just a stupid copypasta

# cumulative maximum can also be thought of as progressively climbing up each largest value sequentially, in this way, it can be thought of as the "Climbing Maximum" Or Climax for short.

# Cumulative Minimum, or "CumMin", is the opposite, where you continuously find the minimum in a sequence.

# The average of both the Cummin and Climax at a particular step is called the "Cumulative Insider" value, The operation is therefore named the "Cuminside" operation.

# Dividing a value by its Climax is called the "Cumulative Normal" Operation (CumNorm)

# For complex values, the CumNorm operation requires the "Absolute" value, in complex values, this correlates to the magnitude of the value.

# To find the "Cumulative absolute value" (CumAbs), you need to do a Climax on the Abs of the value.

# One thought experiment is the possibility of shifting the CumNormed value by its Cuminside value.
# This would look like this:

# ```py
# def forward(ex: Complex):
#   CumAbs = Climax(Abs(ex))
#   CumNormal = ex / CumAbs
  
#   CumInside = (Climax(ex) + CumMin(ex))/2
  
#   FixedEx = CumInside + CumNormal
#   return FixedEx
# ```

# By using this algorithm, you can retain the information inside a complex value, while also normalizing it for use in Rotary Memory or as an overly complicated activation function.
