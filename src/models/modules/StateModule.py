# create statefull torch.nn module
import torch
import torch.nn as nn
class StateModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initState(*args, **kwargs)
    def initState(self, *args, **kwargs):
        self.state = torch.zeros(*args, **kwargs)

    def setState(self, state):
        if not self.training:
            if state is None:
                self.state = None
            else:
                self.state = state

    def getState(self):
        if self.state is None:
            return None
        return self.state
    
    def resetState(self):
        self.initState(*self.state.shape)