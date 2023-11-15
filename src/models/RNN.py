from typing import Any

import os, math, gc, importlib
from torch import nn
import torch
from torch.nn import functional as F


method = torch.jit.script_method
method = lambda x: x
module = torch.jit.ScriptModule
module = nn.Module

class Model(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._for = self.forward
        self.forward = lambda args, state=-1, **kwargs: self.state_forward(args,state=state,returnState=True,flattenbatch=False,**kwargs)
        
    def recursiveSetState(self, module, state=None, prefix='', state2=None, mix=0.0):
        if state is None:
            state = self.getState()
        # recursively go through all the modules and if it has a setState method, call it with the state dict
        for name, child in module.named_children():
            
            if hasattr(child, 'setState'):
                if state2 is None:
                    child.setState(state.get(prefix + name, None))
                else:
                    # print((state[prefix + name] - state2[prefix + name]).sum())
                    child.setState(torch.lerp(state[prefix + name], state2[prefix + name], mix))
            
            self.recursiveSetState(child, state, prefix + name + '.', state2=state2, mix=mix)

    def recursiveGetState(self, module, state=None, prefix=''):
        if state is None:
            state = {}
        # recursively go through all the modules and if it has a getState method, call it and add the result to the state dict
        for name, child in module.named_children():
            if hasattr(child, 'getState'):
                state[prefix + name] = child.state.clone() if isinstance(child.state, torch.Tensor) else child.state
            
            self.recursiveGetState(child, state, prefix + name + '.')

        return state
    
    def setState(self, state, state2 = None, mix = 0.0):
        self.recursiveSetState(self, state, state2=state2, mix=mix)

    def getState(self):
        return self.recursiveGetState(self)
    
    def recursiveResetState(self,m):
        for name, child in m.named_children():
            if hasattr(child, 'resetState'):
                
                child.resetState()
            
            self.recursiveResetState(child)

    def resetState(self):
        self.recursiveResetState(self)

    def state_forward(self, *args, state=-1,flattenbatch=False, returnState=True, full_output=False, **kwargs):
        if state is None:
            state={}

        if state == -1:
            inplace = True
            state = self.getState()

        else:
            inplace = False
            
        logits, state = self._for(*args,state=state, **kwargs)
       
        if not full_output and not self.training :
            
            logits = logits[:,-1,:]
            
        if inplace:
            self.setState(state)
            return logits
        return logits, state
        

    
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


# Pytorch lightning
try:
    class LightningModel(Model):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

 

        
except:
    pass
import torch








    